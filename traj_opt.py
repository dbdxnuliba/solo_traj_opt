from constants import *
from utils import (
    legs,
    derive_skew_ca,
    derive_rot_mat_ca,
    derive_homog_ca,
    derive_reverse_homog_ca,
    derive_mult_homog_point_ca,
    extract_state_ca,
)
from draw import animate_traj
from generate_reference import generate_reference
import numpy as np
import casadi as ca

if __name__ == "__main__":
    skew_ca = derive_skew_ca()
    rot_mat_ca = derive_rot_mat_ca()
    homog_ca = derive_homog_ca()
    reverse_homog_ca = derive_reverse_homog_ca()
    mult_homog_point_ca = derive_mult_homog_point_ca()

    X_ref, U_ref, dt = generate_reference()
    animate_traj(X_ref, U_ref, dt)

    N = N = X_ref.shape[1] - 1

    opti = ca.Opti()
    X = opti.variable(18, N + 1)
    U = opti.variable(24, N + 1)
    J = ca.MX(1, 1)

    for k in range(N + 1):
        # extract state
        p, R, pdot, omega, p_i, f_i = extract_state_ca(X, U, k)
        if k != N:
            (
                p_next,
                R_next,
                pdot_next,
                omega_next,
                p_i_next,
                f_i_next,
            ) = extract_state_ca(X, U, k + 1)
        else:
            p_next, R_next, pdot_next, omega_next, p_i_next, f_i_next = (
                None,
                None,
                None,
                None,
                None,
                None,
            )

        # extract reference
        p_ref, R_ref, pdot_ref, omega_ref, p_i_ref, f_i_ref = extract_state_ca(
            X_ref, U_ref, k
        )

        # objective function
        J += ca.dot(Q_p * (p - p_ref), (p - p_ref))
        for leg in legs:
            J += ca.dot(Q_p_i * (p_i[leg] - p_i_ref[leg]), (p_i[leg] - p_i_ref[leg]))
        J += ca.trace(Gp - Gp @ R_ref.T @ R)
        J += ca.dot(R_pdot * pdot, pdot)
        J += ca.dot(R_omega * omega, omega)
        for leg in legs:
            J += ca.dot(R_f_i * f_i[leg], f_i[leg])

        # dynamics constraints
        f = ca.MX(3, 1)
        tau = ca.MX(3, 1)
        for leg in legs:
            f += f_i[leg]
            tau += ca.cross(p_i[leg] - p, f_i[leg])
        if k != N:
            opti.subject_to(p_next == p + pdot * dt)
            opti.subject_to(pdot_next == pdot + (f / m + g) * dt)
            opti.subject_to(R_next == R @ rot_mat_ca(omega, dt))
            opti.subject_to(
                omega_next
                == omega + B_I_inv @ (R.T @ tau - skew_ca(omega) @ B_I @ omega) * dt
            )

        # kinematics constraints
        T_B = homog_ca(p, R)
        for leg in legs:
            T_Bi = T_B @ B_T_Bi[leg]
            Bi_T = reverse_homog_ca(T_Bi)
            Bi_p_i = mult_homog_point_ca(Bi_T, p_i[leg])
            if leg == legs.FL or leg == legs.FR:
                opti.subject_to(opti.bounded(-x_kin_in_lim, Bi_p_i[0], x_kin_out_lim))
            else:
                opti.subject_to(opti.bounded(-x_kin_out_lim, Bi_p_i[0], x_kin_in_lim))
            opti.subject_to(opti.bounded(-eps, Bi_p_i[1], eps))
            opti.subject_to(opti.bounded(z_kin_lower_lim, Bi_p_i[2], z_kin_upper_lim))

        # friction pyramid constraints
        for leg in legs:
            opti.subject_to(f_i[leg][2] >= 0.0)
            opti.subject_to(
                opti.bounded(-mu * f_i[leg][2], f_i[leg][0], mu * f_i[leg][2])
            )
            opti.subject_to(
                opti.bounded(-mu * f_i[leg][2], f_i[leg][1], mu * f_i[leg][2])
            )

        # contact constraints
        for leg in legs:
            opti.subject_to(p_i[leg][2] >= 0.0)
            opti.subject_to(opti.bounded(-eps, f_i[leg][2] * p_i[leg][2], eps))
            if k != N:
                opti.subject_to(
                    opti.bounded(
                        -eps, f_i[leg][2] * (p_i_next[leg][0] - p_i[leg][0]), eps
                    )
                )
                opti.subject_to(
                    opti.bounded(
                        -eps, f_i[leg][2] * (p_i_next[leg][1] - p_i[leg][1]), eps
                    )
                )

    opti.minimize(J)

    # initial conditions constraint
    opti.subject_to(X[:, 0] == X_ref[:, 0])
    opti.subject_to(U[:, 0] == U_ref[:, 0])

    # initial solution guess
    opti.set_initial(X, X_ref)
    opti.set_initial(U, U_ref)

    # solve NLP
    p_opts = {}
    s_opts = {"print_level": 5}
    opti.solver("ipopt", p_opts, s_opts)
    sol = opti.solve()

    X_sol = np.array(sol.value(X))
    U_sol = np.array(sol.value(U))
    animate_traj(X_sol, U_sol, dt)
