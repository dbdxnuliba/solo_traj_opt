import numpy as np
import scipy.sparse as sp
import osqp

from constants import *
from draw import animate_traj
from utils import extract_state_np, produce_XR_from_X, skew_sp, extract_X_from_XR
from generate_reference import generate_reference


def calc_obj_t(r_ref, l_ref, k_ref, r_cqp, l_cqp, k_cqp):
    P_t = sp.lil_matrix((dim_x_fqp, dim_x_fqp))
    diag_P_t = np.hstack(
        (
            np.tile(phi_r + L_r, 3),
            np.tile(phi_l + L_l, 3),
            np.tile(phi_k + L_k, 3),
            np.tile(psi_f, 12),
        )
    )
    P_t.setdiag(diag_P_t)

    q_t = np.hstack(
        (
            -phi_r * r_ref - L_r * r_cqp,
            -phi_l * l_ref - L_l * l_cqp,
            -phi_k * k_ref - L_k * k_cqp,
            np.zeros(12),
        )
    )

    return P_t, q_t


def calc_dyn_t(l_i_cqp, dt):
    A_dyn_t = sp.lil_matrix((dim_dyn_fqp, dim_x_fqp * 2))
    A_dyn_t[:, :9] = sp.identity(9)
    A_dyn_t[:, 21:30] = -1.0 * sp.identity(9)
    A_dyn_t[0:3, 24:27] = dt / m * sp.identity(3)

    for leg in legs:
        A_dyn_t[3:6, 30 + 3 * leg.value : 30 + 3 * (leg.value + 1)] = dt * sp.identity(
            3
        )
        A_dyn_t[6:9, 30 + 3 * leg.value : 30 + 3 * (leg.value + 1)] = dt * skew_sp(
            l_i_cqp[leg]
        )

    l_dyn_t = np.zeros(dim_dyn_fqp)
    l_dyn_t[5] = m * g * dt
    u_dyn_t = l_dyn_t

    return A_dyn_t, l_dyn_t, u_dyn_t


def calc_fric_t(c_i_cqp):
    A_fric_t = sp.lil_matrix((dim_fric_fqp, dim_x_fqp))
    block = sp.lil_matrix((5, 3))
    block[0, 0] = -1.0
    block[0, 2] = mu
    block[1, 0] = 1.0
    block[1, 2] = mu
    block[2, 1] = -1.0
    block[2, 2] = mu
    block[3, 1] = 1.0
    block[3, 2] = mu
    block[4, 2] = 1.0

    for leg in legs:
        A_fric_t[
            5 * leg.value : 5 * (leg.value + 1),
            9 + 3 * leg.value : 9 + 3 * (leg.value + 1),
        ] = block

    l_fric_t = np.zeros(dim_fric_fqp)

    u_fric_t = np.full(dim_fric_fqp, np.inf)
    for leg in legs:
        if c_i_cqp[leg] == False:
            u_fric_t[4 + 5 * leg.value] = 0.0

    return A_fric_t, l_fric_t, u_fric_t


def calc_kin_t(p_i_cqp):
    A_kin_t = sp.lil_matrix((dim_kin_fqp, dim_x_fqp))
    block = np.vstack(
        (
            [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
            [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
        )
    ).T

    for leg in legs:
        A_kin_t[8 * leg.value : 8 * (leg.value + 1), :3] = block

    l_kin_t = np.full(dim_kin_fqp, -np.inf)

    u_kin_t = np.full(dim_kin_fqp, L_kin)
    for leg in legs:
        u_kin_t[8 * leg.value : 8 * (leg.value + 1)] += block @ p_i_cqp[leg]

    return A_kin_t, l_kin_t, u_kin_t


def solve_force_qp(X_cqp, X_ref, dt):
    N = X_cqp.shape[1] - 1  # number of time intervals
    assert N == X_ref.shape[1] - 1

    # construct objective
    P = sp.lil_matrix((dim_x_fqp * (N + 1), dim_x_fqp * (N + 1)))
    q = np.zeros(dim_x_fqp * (N + 1))
    for idx, t in enumerate(np.arange(N + 1)):
        r_cqp_t, l_cqp_t, k_cqp_t, p_i_cqp_t, f_i_cqp_t = extract_state_np(X_cqp, t)
        r_ref_t, l_ref_t, k_ref_t, p_i_ref_t, f_i_ref_t = extract_state_np(X_ref, t)
        P_t, q_t = calc_obj_t(r_ref_t, l_ref_t, k_ref_t, r_cqp_t, l_cqp_t, k_cqp_t)

        row_start = dim_x_fqp * idx
        row_end = dim_x_fqp * (idx + 1)
        col_start = dim_x_fqp * idx
        col_end = dim_x_fqp * (idx + 1)

        P[row_start:row_end, col_start:col_end] = P_t
        q[row_start:row_end] = q_t

    # construct constraints
    num_constraints = dim_dyn_fqp * N + dim_fric_fqp * (N + 1) + dim_kin_fqp * (N + 1)
    assert num_constraints == 61 * N + 52
    A = sp.lil_matrix((num_constraints, dim_x_fqp * (N + 1)))
    l = np.empty(num_constraints)
    u = np.empty(num_constraints)

    # dynamics
    for idx, t in enumerate(np.arange(1, N + 1)):
        r_cqp_t, l_cqp_t, k_cqp_t, p_i_cqp_t, f_i_cqp_t = extract_state_np(X_cqp, t)
        l_i_cqp_t = {}
        for leg in legs:
            l_i_cqp_t[leg] = p_i_cqp_t[leg] - r_cqp_t
        A_dyn_t, l_dyn_t, u_dyn_t = calc_dyn_t(l_i_cqp_t, dt)

        row_start = dim_dyn_fqp * idx
        row_end = dim_dyn_fqp * (idx + 1)
        col_start = dim_x_fqp * idx
        col_end = dim_x_fqp * (idx + 2)

        A[row_start:row_end, col_start:col_end] = A_dyn_t
        l[row_start:row_end] = l_dyn_t
        u[row_start:row_end] = u_dyn_t

    # friction
    for idx, t in enumerate(np.arange(N + 1)):
        r_cqp_t, l_cqp_t, k_cqp_t, p_i_cqp_t, f_i_cqp_t = extract_state_np(X_cqp, t)
        c_i_cqp_t = {}
        for leg in legs:
            c_i_cqp_t[leg] = p_i_cqp_t[leg][-1] < eps_contact
        A_fric_t, l_fric_t, u_fric_t = calc_fric_t(c_i_cqp_t)

        row_start = dim_dyn_fqp * N + dim_fric_fqp * idx
        row_end = dim_dyn_fqp * N + dim_fric_fqp * (idx + 1)
        col_start = dim_x_fqp * idx
        col_end = dim_x_fqp * (idx + 1)

        A[row_start:row_end, col_start:col_end] = A_fric_t
        l[row_start:row_end] = l_fric_t
        u[row_start:row_end] = u_fric_t

    # kinematics
    for idx, t in enumerate(np.arange(N + 1)):
        r_cqp_t, l_cqp_t, k_cqp_t, p_i_cqp_t, f_i_cqp_t = extract_state_np(X_cqp, t)
        A_kin_t, l_kin_t, u_kin_t = calc_kin_t(p_i_cqp_t)

        row_start = dim_dyn_fqp * N + dim_fric_fqp * (N + 1) + dim_kin_fqp * idx
        row_end = dim_dyn_fqp * N + dim_fric_fqp * (N + 1) + dim_kin_fqp * (idx + 1)
        col_start = dim_x_fqp * idx
        col_end = dim_x_fqp * (idx + 1)

        A[row_start:row_end, col_start:col_end] = A_kin_t
        l[row_start:row_end] = l_kin_t
        u[row_start:row_end] = u_kin_t

    # set up QP and solve
    qp = osqp.OSQP()
    qp.setup(P=P.tocsc(), q=q, A=A.tocsc(), l=l, u=u, **osqp_settings)
    results = qp.solve()

    X_sol_fqp = results.x.reshape((dim_x_fqp, N + 1), order="F")

    X_sol = np.empty((dim_x, N + 1))
    X_sol[:9, :] = X_sol_fqp[:9, :]  # r, l, k
    X_sol[9:21, :] = X_cqp[9:21, :]  # p_i
    X_sol[21:, :] = X_sol_fqp[9:, :]  # f_i

    info = results.info

    return X_sol, info


# test functions
if __name__ == "__main__":
    XR_ref, dt = generate_reference()
    X_ref = extract_X_from_XR(XR_ref)
    X_cqp = X_ref

    X_sol, info = solve_force_qp(X_cqp, X_ref, dt)

    R_init_flat = XR_ref[-9:, 0]
    R_init = np.reshape(R_init_flat, (3, 3), order="F")
    XR_sol = produce_XR_from_X(X_sol, dt, R_init)

    animate_traj(XR_sol, dt)

    import ipdb

    ipdb.set_trace()
