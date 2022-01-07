from constants import *
from draw import animate_traj
from utils import legs, rot_mat_np, extract_state_np, flatten_state_np
import numpy as np


def generate_reference():
    tf = 20.0
    N = int(tf * 10)
    dt = tf / (N)
    t_vals = np.linspace(0, tf, N + 1)

    X = np.zeros((18, N + 1))
    U = np.zeros((24, N + 1))

    for k in range(N + 1):
        t = t_vals[k]
        p = np.array([0.3 * np.sin(t / tf * 6 * np.pi), 0.0, 0.2])
        R = rot_mat_np(np.array([0.0, 1.0, 0.0]), 0.0)
        pdot = np.array([0.0, 0.0, 0.0])
        omega = np.array([0.0, 0.0, 0.0])
        p_i = {}
        f_i = {}
        for leg in legs:
            p_i[leg] = B_p_Bi[leg].copy()
            p_i[leg][0] += 0.3 * np.sin(t / tf * 6 * np.pi)
            if leg == legs.FL or leg == legs.HR:
                p_i[leg][2] += max(0.0, 0.05 * np.sin(10.0 * t))
            else:
                p_i[leg][2] += max(0.0, 0.05 * np.sin(10.0 * t + np.pi))
            f_i[leg] = np.array([0.0, 0.0, 0.0])
            if p_i[leg][2] <= eps:
                f_i[leg][2] = m * np.linalg.norm(g) / 4.0
        X[:, k], U[:, k] = flatten_state_np(p, R, pdot, omega, p_i, f_i)

    return X, U, dt


if __name__ == "__main__":

    X, U, dt = generate_reference()
    animate_traj(X, U, dt)
