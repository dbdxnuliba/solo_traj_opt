import numpy as np
import scipy.sparse as sp
import osqp

from constants import *
from utils import extract_state_np
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


# test functions
if __name__ == "__main__":
    XR, dt = generate_reference()
    N = XR.shape[1] - 1

    for t in np.arange(N + 1):
        r_ref, l_ref, k_ref, p_i_ref, f_i_ref, R_ref = extract_state_np(XR, t)
        P_t, q_t = calc_obj_t(r_ref, l_ref, k_ref, r_ref, l_ref, k_ref)
        import ipdb

        ipdb.set_trace()
