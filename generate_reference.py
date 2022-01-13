from constants import *
from draw import animate_traj
from utils import (
    legs,
    mult_homog_point_np,
    rot_mat_2d_np,
    rot_mat_np,
    flatten_state_np,
    homog_np,
)
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import CubicHermiteSpline


# linearly interpolate x and y, evaluate at t
def linear_interp_t(x, y, t):
    f = interp1d(x, y)
    return f(t)


# interpolate x and y, evaluate at t using cubic splines with zero deriatives
# this creates an interpolation similar to linear interpolation, but with
# smoothed corners
def cubic_interp_t(x, y, t):
    f = CubicHermiteSpline(x, y, np.zeros_like(x))
    return f(t)


# sinusoidal function evaluated at t defined using oscillation period, minimum
# and maximum values
def sinusoid(period, min_val, max_val, t):
    return (max_val - min_val) / 2.0 * (1 - np.cos(2 * np.pi / period * t)) + min_val


def generate_reference():
    tf = 5.0
    N = int(tf * 20)
    dt = tf / (N)
    t_vals = np.linspace(0, tf, N + 1)

    X = np.zeros((18, N + 1))
    U = np.zeros((24, N + 1))

    for k in range(N + 1):
        angle = cubic_interp_t(
            [0, 1.6, 2.5, 3.4, tf], [0, 0, np.pi / 2.0, 0, 0], t_vals[k]
        )
        p = np.array([-l_Bx / 2.0, 0.0, l_thigh])
        p_xz = rot_mat_2d_np(angle) @ np.array([l_Bx / 2.0, 0.0])
        p += np.array([p_xz[0], 0.0, p_xz[1]])
        R = rot_mat_np(np.array([0.0, 1.0, 0.0]), -angle)
        pdot = np.array([0.0, 0.0, 0.0])
        omega = np.array([0.0, 0.0, 0.0])
        p_i = {}
        f_i = {}
        T_B = homog_np(p, R)
        for leg in legs:
            if leg == legs.FL or leg == legs.FR:
                p_Bi = mult_homog_point_np(T_B, B_p_Bi[leg])
                p_i[leg] = p_Bi.copy()
                p_i[leg][2] -= l_thigh
            else:
                p_i[leg] = B_p_Bi[leg].copy()
            f_i[leg] = np.array([0.0, 0.0, 0.0])
            if p_i[leg][2] <= eps:
                f_i[leg][2] = m * np.linalg.norm(g) / 4.0
        X[:, k], U[:, k] = flatten_state_np(p, R, pdot, omega, p_i, f_i)

    return X, U, dt


if __name__ == "__main__":

    X, U, dt = generate_reference()
    animate_traj(X, U, dt)
