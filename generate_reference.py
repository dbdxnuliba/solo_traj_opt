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
    motion_type = "stand"

    if motion_type == "trot":
        tf = 10.0
    if motion_type == "bound":
        tf = 10.0
    if motion_type == "pronk":
        tf = 10.0
    if motion_type == "jump":
        tf = 2.0
    elif motion_type == "stand":
        tf = 5.0
    elif motion_type == "front-hop":
        tf = 5.0

    N = int(tf * 50)
    dt = tf / (N)
    t_vals = np.linspace(0, tf, N + 1)

    X = np.zeros((18, N + 1))
    U = np.zeros((24, N + 1))

    for k in range(N + 1):
        if motion_type == "trot":
            if k * dt < 0.5 or k * dt > 9.5:
                body_x = -0.3
            else:
                body_x = sinusoid(
                    period=9.0,
                    min_val=-0.3,
                    max_val=0.3,
                    t=t_vals[k],
                    phase_offset=-2 * np.pi / 9.0 * 0.5,
                )
            p = np.array([body_x, 0.0, 0.25])
            R = np.eye(3)
            p_i = {}
            for leg in legs:
                p_i[leg] = B_p_Bi[leg].copy()
                p_i[leg][0] += body_x
                if k * dt < 0.5 or k * dt > 9.5:
                    pass
                else:
                    if leg == legs.FL or leg == legs.HR:
                        p_i[leg][2] += max(
                            0.0, sinusoid(0.5, -0.1, 0.1, t_vals[k], pi / 2.0)
                        )
                    else:
                        p_i[leg][2] += max(
                            0.0, sinusoid(0.5, -0.1, 0.1, t_vals[k], 3.0 * pi / 2.0)
                        )
        if motion_type == "bound":
            if k * dt < 0.5 or k * dt > 9.5:
                body_x = -0.3
            else:
                body_x = sinusoid(
                    period=9.0,
                    min_val=-0.3,
                    max_val=0.3,
                    t=t_vals[k],
                    phase_offset=-2 * np.pi / 9.0 * 0.5,
                )
            p = np.array([body_x, 0.0, 0.25])
            R = np.eye(3)
            p_i = {}
            for leg in legs:
                p_i[leg] = B_p_Bi[leg].copy()
                p_i[leg][0] += body_x
                if k * dt < 0.5 or k * dt > 9.5:
                    pass
                else:
                    if leg == legs.FL or leg == legs.FR:
                        p_i[leg][2] += max(
                            0.0, sinusoid(0.5, -0.1, 0.1, t_vals[k], pi / 2.0)
                        )
                    else:
                        p_i[leg][2] += max(
                            0.0, sinusoid(0.5, -0.1, 0.1, t_vals[k], 3.0 * pi / 2.0)
                        )
        if motion_type == "pronk":
            if k * dt < 0.5 or k * dt > 9.5:
                body_x = -0.3
            else:
                body_x = sinusoid(
                    period=9.0,
                    min_val=-0.3,
                    max_val=0.3,
                    t=t_vals[k],
                    phase_offset=-2 * np.pi / 9.0 * 0.5,
                )
            p = np.array([body_x, 0.0, 0.25])
            R = np.eye(3)
            p_i = {}
            for leg in legs:
                p_i[leg] = B_p_Bi[leg].copy()
                p_i[leg][0] += body_x
                if k * dt < 0.5 or k * dt > 9.5:
                    pass
                else:
                    p_i[leg][2] += max(
                        0.0, sinusoid(0.5, -0.1, 0.1, t_vals[k], pi / 2.0)
                    )
        if motion_type == "jump":
            t_apex = 0.3
            z_apex = np.linalg.norm(g) * t_apex**2 / 2.0
            body_z = cubic_interp_t(
                [0, 0.2 * tf, 0.2 * tf + t_apex, 0.2 * tf + 2 * t_apex, tf],
                [0, 0, z_apex, 0, 0],
                t_vals[k],
            )
            p = np.array([0.0, 0.0, 0.2])
            p[2] += body_z
            R = np.eye(3)
            p_i = {}
            for leg in legs:
                p_i[leg] = B_p_Bi[leg].copy()
                p_i[leg][2] += body_z
        elif motion_type == "stand":
            body_height = 0.2
            angle = cubic_interp_t(
                [0, 1.6, 2.5, 3.4, tf], [0, 0, np.pi / 2.0, 0, 0], t_vals[k]
            )
            p = np.array([-l_Bx / 2.0, 0.0, body_height])
            p_xz = rot_mat_2d_np(angle) @ np.array([l_Bx / 2.0, 0.0])
            p += np.array([p_xz[0], 0.0, p_xz[1]])
            R = rot_mat_np(np.array([0.0, 1.0, 0.0]), -angle)
            p_i = {}
            T_B = homog_np(p, R)
            for leg in legs:
                if leg == legs.FL or leg == legs.FR:
                    p_Bi = mult_homog_point_np(T_B, B_p_Bi[leg])
                    p_i[leg] = p_Bi.copy()
                    p_i[leg][2] -= body_height
                else:
                    p_i[leg] = B_p_Bi[leg].copy()
        elif motion_type == "front-hop":
            body_height = 0.2
            angle = cubic_interp_t(
                [0, 2.0, 2.5, 3.0, tf], [0, 0, np.pi / 4.0, 0, 0], t_vals[k]
            )
            p = np.array([-l_Bx / 2.0, 0.0, body_height])
            p_xz = rot_mat_2d_np(angle) @ np.array([l_Bx / 2.0, 0.0])
            p += np.array([p_xz[0], 0.0, p_xz[1]])
            R = rot_mat_np(np.array([0.0, 1.0, 0.0]), -angle)
            p_i = {}
            T_B = homog_np(p, R)
            for leg in legs:
                if leg == legs.FL or leg == legs.FR:
                    p_Bi = mult_homog_point_np(T_B, B_p_Bi[leg])
                    p_i[leg] = p_Bi.copy()
                    p_i[leg][2] -= body_height
                else:
                    p_i[leg] = B_p_Bi[leg].copy()

        pdot = np.array([0.0, 0.0, 0.0])
        omega = np.array([0.0, 0.0, 0.0])
        f_i = {}
        for leg in legs:
            f_i[leg] = np.array([0.0, 0.0, 0.0])
            if p_i[leg][2] <= eps:
                f_i[leg][2] = m * np.linalg.norm(g) / 4.0
        X[:, k], U[:, k] = flatten_state_np(p, R, pdot, omega, p_i, f_i)

    return X, U, dt


if __name__ == "__main__":

    X, U, dt = generate_reference()
    animate_traj(X, U, dt, repeat=False)
