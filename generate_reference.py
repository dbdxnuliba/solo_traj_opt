from constants import *
from draw import animate_traj
from utils import (
    flatten_state_and_rot_np,
    legs,
    mult_homog_point_np,
    rot_mat_2d_np,
    rot_mat_np,
    flatten_state_and_rot_np,
    homog_np,
)
import numpy as np
from numpy import pi
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
def sinusoid(period, min_val, max_val, t, phase_offset=0):
    return (max_val - min_val) / 2.0 * (
        1 - np.cos(2 * np.pi / period * t + phase_offset)
    ) + min_val


def generate_reference():
    motion_type = "walk"

    if motion_type == "walk":
        tf = 5.0
    if motion_type == "jump":
        tf = 2.0
    elif motion_type == "stand":
        tf = 5.0

    N = int(tf * 100)
    dt = tf / (N)
    t_vals = np.linspace(0, tf, N + 1)

    XR = np.zeros((dim_xR, N + 1))

    for t in range(N + 1):
        if motion_type == "walk":
            body_x = sinusoid(period=2.5, min_val=-0.1,
                              max_val=0.1, t=t_vals[t])
            r = np.array([body_x, 0.0, 0.2])
            R = np.eye(3)
            p_i = {}
            for leg in legs:
                p_i[leg] = B_p_Bi[leg].copy()
                p_i[leg][0] += body_x
                if leg == legs.FL or leg == legs.HR:
                    p_i[leg][2] += max(0.0, sinusoid(0.6, -
                                       0.05, 0.05, t_vals[t], 0.0))
                else:
                    p_i[leg][2] += max(0.0, sinusoid(0.6, -
                                       0.05, 0.05, t_vals[t], pi))
        if motion_type == "jump":
            t_apex = 0.3
            z_apex = g * t_apex**2 / 2.0
            body_z = cubic_interp_t(
                [0, 0.2 * tf, 0.2 * tf + t_apex, 0.2 * tf + 2 * t_apex, tf],
                [0, 0, z_apex, 0, 0],
                t_vals[t],
            )
            r = np.array([0.0, 0.0, 0.2])
            r[2] += body_z
            R = np.eye(3)
            p_i = {}
            for leg in legs:
                p_i[leg] = B_p_Bi[leg].copy()
                p_i[leg][2] += body_z
        elif motion_type == "stand":
            angle = cubic_interp_t(
                [0, 1.6, 2.5, 3.4, tf], [0, 0, np.pi / 2.0, 0, 0], t_vals[t]
            )
            r = np.array([-l_Bx / 2.0, 0.0, l_thigh])
            r_xz = rot_mat_2d_np(angle) @ np.array([l_Bx / 2.0, 0.0])
            r += np.array([r_xz[0], 0.0, r_xz[1]])
            R = rot_mat_np(np.array([0.0, 1.0, 0.0]), -angle)
            p_i = {}
            T_B = homog_np(r, R)
            for leg in legs:
                if leg == legs.FL or leg == legs.FR:
                    p_Bi = mult_homog_point_np(T_B, B_p_Bi[leg])
                    p_i[leg] = p_Bi.copy()
                    p_i[leg][2] -= l_thigh
                else:
                    p_i[leg] = B_p_Bi[leg].copy()

        l = np.array([0.0, 0.0, 0.0])
        k = np.array([0.0, 0.0, 0.0])
        f_i = {}
        for leg in legs:
            f_i[leg] = np.array([0.0, 0.0, 0.0])
            if p_i[leg][2] <= eps:
                f_i[leg][2] = m * g / 4.0

        XR[:, t] = flatten_state_and_rot_np(r, l, k, p_i, f_i, R)

    return XR, dt


if __name__ == "__main__":

    XR, dt = generate_reference()
    animate_traj(XR, dt)
