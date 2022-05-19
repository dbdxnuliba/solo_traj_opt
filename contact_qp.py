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


# test functions
if __name__ == "__main__":
    XR_fqp, dt = generate_reference()
    X_fqp = extract_X_from_XR(XR_fqp)

    N = X_fqp.shape[1] - 1  # number of time intervals

    for idx, t in enumerate(np.arange(1, N + 1)):
        r_fqp_t, l_fqp_t, k_fqp_t, p_i_fqp_t, f_i_fqp_t = extract_state_np(X_fqp, t)
        c_i_fqp_t = {}
        for leg in legs:
            c_i_fqp_t[leg] = p_i_fqp_t[leg][-1] < eps_contact
        (
            _,
            _,
            k_fqp_t_minus,
            _,
            _,
        ) = extract_state_np(X_fqp, t - 1)
        dk_fqp_t = k_fqp_t - k_fqp_t_minus
        P_t, q_t = calc_obj_t(r_fqp_t, l_fqp_t, p_i_fqp_t)
        A_dyn_t, l_dyn_t, u_dyn_t = calc_dyn_t(f_i_fqp_t, dk_fqp_t, dt)
        A_loc_t, l_loc_t, u_loc_t = calc_loc_t(p_i_fqp_t, c_i_fqp_t)
        A_kin_t, l_kin_t, u_kin_t = calc_kin_t()

        import ipdb

        ipdb.set_trace()
