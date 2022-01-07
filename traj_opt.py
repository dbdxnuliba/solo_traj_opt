from constants import *
from utils import (
    derive_skew,
    derive_rotMat,
    derive_homog,
    derive_reverse_homog,
    derive_mult_homog_point,
)
import numpy as np
import casadi as ca

# temporary hard coded constant desired values
tf = 5
N = int(tf * 4)
dt = tf / (N)

p_des = np.array([0.0, 0.0, l_thigh / 2.0])
p_i_des = {}
for leg in legs:
    p_i_des[leg] = B_p_Bi[leg]
R_des = np.eye(3)

skew_func = derive_skew()
rotMat_func = derive_rotMat()
homog_func = derive_homog()
reverse_homog_func = derive_reverse_homog()
mult_homog_point_func = derive_mult_homog_point()


def extract_state(X, U, k):
    p = X[:3, k]
    R_flat = X[3:12, k]
    R = ca.reshape(R_flat, 3, 3)
    pdot = X[12:15, k]
    omega = X[15:18, k]
    p_i = {}
    f_i = {}
    for leg in legs:
        p_i[leg] = U[3 * leg.value : leg.value * 3 + 3, k]
        f_i[leg] = U[12 + 3 * leg.value : 12 + leg.value * 3 + 3, k]
    return p, R, pdot, omega, p_i, f_i


def flatten_state(p, R, pdot, omega, p_i, f_i):
    R_flat = ca.reshape(R, 9, 1)
    p_i_flat = ca.MX(12, 1)
    f_i_flat = ca.MX(12, 1)
    for leg in legs:
        p_i_flat[3 * leg.value : leg.value * 3 + 3] = p_i[leg]
        f_i_flat[3 * leg.value : leg.value * 3 + 3] = f_i[leg]

    X_k = ca.vertcat(p, R_flat, pdot, omega)
    U_k = ca.vertcat(p_i_flat, f_i_flat)

    return X_k, U_k


if __name__ == "__main__":
    opti = ca.Opti()
    X = opti.variable(18, N + 1)
    U = opti.variable(24, N + 1)
    J = ca.MX(1, 1)

    for k in range(N + 1):
        # extract state
        p, R, pdot, omega, p_i, f_i = extract_state(X, U, k)
        if k != N:
            p_next, R_next, pdot_next, omega_next, p_i_next, f_i_next = extract_state(
                X, U, k + 1
            )
        else:
            p_next, R_next, pdot_next, omega_next, p_i_next, f_i_next = (
                None,
                None,
                None,
                None,
                None,
                None,
            )

        # objective function
        J += ca.dot(Q_p * (p - p_des), (p - p_des))
        for leg in legs:
            J += ca.dot(Q_p_i * (p_i[leg] - p_i_des[leg]), (p_i[leg] - p_i_des[leg]))
        J += ca.trace(Gp - Gp @ R_des.T @ R)
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
            opti.subject_to(R_next == R @ rotMat_func(omega, dt))
            opti.subject_to(
                omega_next
                == omega + B_I_inv @ (R.T @ tau - skew_func(omega) @ B_I @ omega) * dt
            )

        # kinematics constraints
        T_B = homog_func(p, R)
        for leg in legs:
            T_Bi = T_B @ B_T_Bi[leg]
            Bi_T = reverse_homog_func(T_Bi)
            Bi_p_i = mult_homog_point_func(Bi_T, p_i[leg])
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
            opti.subject_to(f_i[leg][2] * p_i[leg][2] < eps)
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

    # initial and final conditions constraint
    # TODO

    # initial solution guess
    # TODO
