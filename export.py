import numpy as np
from scipy.spatial.transform import Rotation

from constants import *
from generate_reference import generate_reference
from utils import extract_state_np, solo_IK_np, solo_jac_transpose_np


def export_to_csv(X, U, dt, fname):
    N = X.shape[1] - 1

    to_save = np.zeros((N + 1, 38))
    for k in range(N + 1):
        # extract state variables
        p, R, pdot, omega, p_i, f_i = extract_state_np(X, U, k)
        if k != N:
            (
                p_next,
                R_next,
                pdot_next,
                omega_next,
                p_i_next,
                f_i_next,
            ) = extract_state_np(X, U, k + 1)
        else:
            p_next, R_next, pdot_next, omega_next, p_i_next, f_i_next = (
                None,
                None,
                None,
                None,
                None,
                None,
            )
        t = k * dt

        # convert orientation to quaternion
        quat_xyzw = Rotation.from_matrix(R).as_quat()
        quat = np.roll(quat_xyzw, 1)

        # calculate joint angle q
        q_i = solo_IK_np(p, R, p_i)
        q = []
        for leg in legs:
            q = np.hstack((q, q_i[leg]))

        # calculate joint velocity qdot
        if k != N:
            q_i_next = solo_IK_np(p_next, R_next, p_i_next)
            q_next = []
            for leg in legs:
                q_next = np.hstack((q_next, q_i_next[leg]))
            qdot = (q_next - q) / dt
        else:
            qdot = np.zeros_like(q)

        # calculate joint torque tau
        # TODO: jacobian transpose calculation from f_i
        tau_i = solo_jac_transpose_np(p, R, p_i, f_i)
        tau = []
        for leg in legs:
            tau = np.hstack((tau, tau_i[leg]))

        # note the reverse signs in joint variables to make it consistent
        # with RL and robot control code
        traj_t = np.hstack((t, p, quat, pdot, omega, -q, -qdot, -tau))
        to_save[k, :] = traj_t

    np.savetxt("csv/" + fname + ".csv", to_save, delimiter=", ", fmt="%0.16f")


if __name__ == "__main__":
    X_ref, U_ref, dt = generate_reference()
    export_to_csv(X_ref, U_ref, dt, "test")
