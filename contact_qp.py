import numpy as np
import scipy.sparse as sp
import osqp

from constants import *
from draw import animate_traj
from utils import extract_state_np, skew_sp, extract_X_from_XR
from generate_reference import generate_reference


def calc_obj_t(r_fqp, l_fqp, p_i_fqp):
    P_t = sp.lil_matrix((dim_x_cqp, dim_x_cqp))
    diag_P_t = np.hstack(
        (
            np.tile(L_r, 3),
            np.tile(L_l, 3),
            np.tile(L_p, 12),
        )
    )
    P_t.setdiag(diag_P_t)

    q_t = np.empty(dim_x_cqp)
    q_t[0:3] = -L_r * r_fqp
    q_t[3:6] = -L_l * l_fqp
    for leg in legs:
        q_t[6 + 3 * leg.value : 6 + 3 * (leg.value + 1)] = -L_p * p_i_fqp[leg]

    return P_t, q_t


def calc_dyn_t(f_i_fqp, dk_fqp, dt):
    A_dyn_t = sp.lil_matrix((dim_dyn_cqp, dim_x_cqp * 2))
    A_dyn_t[0:3, 0:3] = sp.identity(3)
    A_dyn_t[0:3, 18:21] = -1.0 * sp.identity(3)
    A_dyn_t[0:3, 21:24] = dt / m * sp.identity(3)

    sum_f_i_cqp = np.zeros(3)
    for leg in legs:
        sum_f_i_cqp += f_i_fqp[leg]
        A_dyn_t[3:6, 24 + 3 * leg.value : 24 + 3 * (leg.value + 1)] = -dt * skew_sp(
            f_i_fqp[leg]
        )
    A_dyn_t[3:6, 18:21] = dt * skew_sp(sum_f_i_cqp)

    l_dyn_t = np.hstack((np.zeros(3), dk_fqp))
    u_dyn_t = l_dyn_t

    return A_dyn_t, l_dyn_t, u_dyn_t


def calc_loc_t(p_i_fqp_t, c_i_fqp_t):
    A_loc_t = sp.lil_matrix((dim_loc_cqp, dim_x_cqp * 2))
    A_loc_t[:, 24:36] = sp.identity(dim_loc_cqp)

    l_loc_t = np.zeros(dim_loc_cqp)
    for leg in legs:
        if c_i_fqp_t[leg] == True:
            A_loc_t[
                3 * leg.value : 3 * leg.value + 2,
                6 + 3 * leg.value : 6 + 3 * leg.value + 2,
            ] = -1.0 * sp.identity(2)
        else:
            l_loc_t[3 * leg.value : 3 * (leg.value + 1)] = p_i_fqp_t[leg]

    u_loc_t = l_loc_t

    return A_loc_t, l_loc_t, u_loc_t


def calc_kin_t():
    A_kin_t = sp.lil_matrix((dim_kin_cqp, dim_x_cqp))
    block = np.vstack(
        (
            [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
            [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
        )
    ).T

    for leg in legs:
        A_kin_t[8 * leg.value : 8 * (leg.value + 1), :3] = block
        A_kin_t[
            8 * leg.value : 8 * (leg.value + 1),
            6 + 3 * leg.value : 6 + 3 * (leg.value + 1),
        ] = (
            -1.0 * block
        )

    l_kin_t = np.full(dim_kin_cqp, -np.inf)
    u_kin_t = np.full(dim_kin_cqp, L_kin)

    return A_kin_t, l_kin_t, u_kin_t


def solve_contact_qp(X_fqp, dt):
    N = X_fqp.shape[1] - 1  # number of time intervals

    # construct objective
    P = sp.lil_matrix((dim_x_cqp * (N + 1), dim_x_cqp * (N + 1)))
    q = np.zeros(dim_x_cqp * (N + 1))
    for idx, t in enumerate(np.arange(N + 1)):
        r_fqp_t, l_fqp_t, k_fqp_t, p_i_fqp_t, f_i_fqp_t = extract_state_np(X_fqp, t)
        P_t, q_t = calc_obj_t(r_fqp_t, l_fqp_t, p_i_fqp_t)

        row_start = dim_x_cqp * idx
        row_end = dim_x_cqp * (idx + 1)
        col_start = dim_x_cqp * idx
        col_end = dim_x_cqp * (idx + 1)

        P[row_start:row_end, col_start:col_end] = P_t
        q[row_start:row_end] = q_t

    # construct constraints
    num_constraints = dim_dyn_cqp * N + dim_loc_cqp * N + dim_kin_cqp * (N + 1)
    assert num_constraints == 50 * N + 32
    A = sp.lil_matrix((num_constraints, dim_x_cqp * (N + 1)))
    l = np.empty(num_constraints)
    u = np.empty(num_constraints)

    # dynamics
    for idx, t in enumerate(np.arange(1, N + 1)):
        r_fqp_t, l_fqp_t, k_fqp_t, p_i_fqp_t, f_i_fqp_t = extract_state_np(X_fqp, t)
        (
            _,
            _,
            k_fqp_t_minus,
            _,
            _,
        ) = extract_state_np(X_fqp, t - 1)
        dk_fqp_t = k_fqp_t - k_fqp_t_minus
        A_dyn_t, l_dyn_t, u_dyn_t = calc_dyn_t(f_i_fqp_t, dk_fqp_t, dt)

        row_start = dim_dyn_cqp * idx
        row_end = dim_dyn_cqp * (idx + 1)
        col_start = dim_x_cqp * idx
        col_end = dim_x_cqp * (idx + 2)

        A[row_start:row_end, col_start:col_end] = A_dyn_t
        l[row_start:row_end] = l_dyn_t
        u[row_start:row_end] = u_dyn_t

    # location
    for idx, t in enumerate(np.arange(1, N + 1)):
        r_fqp_t, l_fqp_t, k_fqp_t, p_i_fqp_t, f_i_fqp_t = extract_state_np(X_fqp, t)
        c_i_fqp_t = {}
        for leg in legs:
            c_i_fqp_t[leg] = p_i_fqp_t[leg][-1] < eps_contact
        A_loc_t, l_loc_t, u_loc_t = calc_loc_t(p_i_fqp_t, c_i_fqp_t)

        row_start = dim_dyn_cqp * N + dim_loc_cqp * idx
        row_end = dim_dyn_cqp * N + dim_loc_cqp * (idx + 1)
        col_start = dim_x_cqp * idx
        col_end = dim_x_cqp * (idx + 2)

        A[row_start:row_end, col_start:col_end] = A_loc_t
        l[row_start:row_end] = l_loc_t
        u[row_start:row_end] = u_loc_t

    # kinematics
    for idx, t in enumerate(np.arange(N + 1)):
        A_kin_t, l_kin_t, u_kin_t = calc_kin_t()

        row_start = dim_dyn_cqp * N + dim_loc_cqp * N + dim_kin_cqp * idx
        row_end = dim_dyn_cqp * N + dim_loc_cqp * N + dim_kin_cqp * (idx + 1)
        col_start = dim_x_cqp * idx
        col_end = dim_x_cqp * (idx + 1)

        A[row_start:row_end, col_start:col_end] = A_kin_t
        l[row_start:row_end] = l_kin_t
        u[row_start:row_end] = u_kin_t

    # set up QP and solve
    qp = osqp.OSQP()
    qp.setup(P=P.tocsc(), q=q, A=A.tocsc(), l=l, u=u, **osqp_settings)
    results = qp.solve()

    X_sol_cqp = results.x.reshape((dim_x_cqp, N + 1), order="F")

    X_sol = np.empty((dim_x, N + 1))
    X_sol[:6, :] = X_sol_cqp[:6, :]  # r, l
    X_sol[6:9, :] = X_fqp[6:9, :]  # k
    X_sol[9:21, :] = X_sol_cqp[6:, :]  # p_i
    X_sol[21:, :] = X_fqp[21:, :]  # f_i

    info = results.info

    return X_sol, info


# test functions
if __name__ == "__main__":
    XR_fqp, dt = generate_reference()
    X_fqp = extract_X_from_XR(XR_fqp)

    X_sol, info = solve_contact_qp(X_fqp, dt)

    # temp
    XR_sol = np.zeros(XR_fqp.shape)
    XR_sol[:dim_x, :] = X_sol
    XR_sol[-9:, :] = XR_fqp[-9:, :]
    animate_traj(XR_sol, dt)

    import ipdb

    ipdb.set_trace()
