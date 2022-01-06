from constants import *
from utils import derive_skew, derive_rotMat, derive_homog, derive_reverse_homog, derive_mult_homog_point
import numpy as np
import casadi as ca

tf = 5
N = int(tf*4)
dt = tf/(N)
epsilon = 1e-6

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
        p_i[leg] = U[3*leg.value: leg.value*3+3, k]
        f_i[leg] = U[12+3*leg.value: 12+leg.value*3+3, k]
    return p, R, pdot, omega, p_i, f_i


if __name__ == "__main__":
    # opti = ca.Opti()
    # X = opti.variable(18, N+1)
    # U = opti.variable(24, N+1)

    # temporarily use SX for debugging
    X = ca.SX.sym('X', 18, N+1)
    U = ca.SX.sym('U', 24, N+1)

    # objective function
    # TODO

    for k in range(N+1):
        # extract state
        p, R, pdot, omega, p_i, f_i = extract_state(X, U, k)
        if k != N:
            p_next, R_next, pdot_next, omega_next, p_i_next, f_i_next = extract_state(
                X, U, k+1)
        else:
            p_next, R_next, pdot_next, omega_next, p_i_next, f_i_next = None, None, None, None, None, None

        # dynamics constraints
        # f = ca.MX(3, 1)
        # tau = ca.MX(3, 1)
        # temporarily use SX for debugging
        f = ca.SX(3, 1)
        tau = ca.SX(3, 1)
        for leg in legs:
            f += f_i[leg]
            tau += ca.cross(p_i[leg], f_i[leg])

        p_next_eqn = p + pdot * dt
        pdot_next_dqn = pdot + (f/m - g) * dt
        R_next_eqn = R @ rotMat_func(omega, dt)
        omega_next_eqn = omega + \
            B_I_inv @ (R.T @ tau - skew_func(omega) @ B_I @ omega) * dt

        # kinematics constraints
        T_B = homog_func(p, R)
        for leg in legs:
            T_Bi = T_B @ B_T_Bi[leg]
            Bi_T = reverse_homog_func(T_Bi)
            Bi_p_i = mult_homog_point_func(Bi_T, p_i[leg])

        # friction cone constraints
        # TODO

        # contact constraints
        # TODO

        # import ipdb
        # ipdb.set_trace()

    # initial and final conditions constraint
    # TODO

    # initial solution guess
    # TODO
