from constants import *
from draw import draw
from utils import legs, rot_mat_np, extract_state_np, flatten_state_np
import numpy as np
import matplotlib.pyplot as plt


def generate_reference():
    tf = 5
    N = int(tf * 4)
    dt = tf / (N)

    X = np.zeros((18, N + 1))
    U = np.zeros((24, N + 1))

    for k in range(N + 1):
        p = np.array([0.0, 0.0, 0.3])
        R = rot_mat_np(np.array([0, 1, 0]), 0.1)
        pdot = np.array([0.0, 0.0, 0.0])
        omega = np.array([0.0, 0.0, 0.0])
        p_i = {}
        f_i = {}
        for leg in legs:
            p_i[leg] = B_p_Bi[leg]
            f_i[leg] = np.array([0.0, 0.0, 0.0])
            if p_i[leg][2] <= eps:
                f_i[leg][2] = m / (4.0 * np.linalg.norm(g))
        X[:, k], U[:, k] = flatten_state_np(p, R, pdot, omega, p_i, f_i)

    return X, U, N, dt


if __name__ == "__main__":

    X, U, N, dt = generate_reference()

    p, R, pdot, omega, p_i, f_i = extract_state_np(X, U, 0)

    draw(p=p, R=R, p_i=p_i, f_i=f_i)
    plt.show()
