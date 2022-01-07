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

tf = 5
N = int(tf * 4)
dt = tf / (N)
eps = 1e-6

# kinematics constraints paramters
x_kin_lb = -l_Bx / 2.0
x_kin_ub = l_thigh
z_kin_lb = -l_thigh
z_kin_ub = l_thigh

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


if __name__ == "__main__":
    opti = ca.Opti()
    X = opti.variable(18, N + 1)
    U = opti.variable(24, N + 1)

    # objective function
    # TODO

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

        # dynamics constraints
        f = ca.MX(3, 1)
        tau = ca.MX(3, 1)
        for leg in legs:
            f += f_i[leg]
            tau += ca.cross(p_i[leg], f_i[leg])
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
            opti.subject_to(opti.bounded(x_kin_lb, Bi_p_i[0], x_kin_ub))
            opti.subject_to(opti.bounded(-eps, Bi_p_i[1], eps))
            opti.subject_to(opti.bounded(z_kin_lb, Bi_p_i[2], z_kin_ub))

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
        # TODO

    # initial and final conditions constraint
    # TODO

    # initial solution guess
    # TODO
