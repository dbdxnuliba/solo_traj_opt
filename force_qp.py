import numpy as np
import scipy.sparse as sp
import osqp

from constants import *
from utils import extract_state_np, skew_sp
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
            u_fric_t[5 * leg.value - 1] = 0.0

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


# test functions
if __name__ == "__main__":
    XR, dt = generate_reference()
    N = XR.shape[1] - 1

    for t in np.arange(N + 1):
        r_ref, l_ref, k_ref, p_i_ref, f_i_ref, R_ref = extract_state_np(XR, t)
        r_cqp = r_ref
        l_cqp = l_ref
        k_cqp = k_ref
        p_i_cqp = p_i_ref
        l_i_cqp = {}
        c_i_cqp = {}
        for leg in legs:
            l_i_cqp[leg] = p_i_cqp[leg] - r_cqp
            c_i_cqp[leg] = p_i_cqp[leg][-1] < eps_contact

        P_t, q_t = calc_obj_t(r_ref, l_ref, k_ref, r_cqp, l_cqp, k_cqp)
        A_dyn_t, l_dyn_t, u_dyn_t = calc_dyn_t(l_i_cqp, dt)
        A_fric_t, l_fric_t, u_fric_t = calc_fric_t(c_i_cqp)
        A_kin_t, l_kin_t, u_kin_t = calc_kin_t(p_i_cqp)

        import ipdb

        ipdb.set_trace()
