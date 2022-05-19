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

    l_dyn_t = np.zeros(9)
    l_dyn_t[5] = m * g * dt
    u_dyn_t = l_dyn_t

    return A_dyn_t, l_dyn_t, u_dyn_t


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
        for leg in legs:
            l_i_cqp[leg] = p_i_cqp[leg] - r_cqp

        P_t, q_t = calc_obj_t(r_ref, l_ref, k_ref, r_cqp, l_cqp, k_cqp)
        A_dyn_t, l_dyn_t, u_dyn_t = calc_dyn_t(l_i_cqp, dt)

        import ipdb

        ipdb.set_trace()
