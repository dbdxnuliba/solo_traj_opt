from constants import *
from utils import homog, mult_homog_point
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.style.use("seaborn")


def draw(p, R, p_i, f_i, f_len=0.5):
    T_B = homog(p, R)
    p_Bi = {}
    for leg in legs:
        p_Bi[leg] = mult_homog_point(T_B, B_p_Bi[leg])

    anim_fig = plt.figure(figsize=(6, 6))
    ax = Axes3D(anim_fig, auto_add_to_figure=False)
    anim_fig.add_axes(ax)
    ax.view_init(azim=-45)
    ax.set_xlim3d([-0.5, 0.5])
    ax.set_ylim3d([-0.5, 0.5])
    ax.set_zlim3d([0, 1])
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if ax.collections:
        ax.collections.pop()

    body_coords = np.vstack(
        (p_Bi[legs.FL], p_Bi[legs.FR], p_Bi[legs.HR], p_Bi[legs.HL], p_Bi[legs.FL])
    ).T
    line = plt.plot([], [])[0]
    line.set_data(body_coords[0], body_coords[1])
    line.set_3d_properties(body_coords[2])
    line.set_color("b")
    line.set_marker("o")

    feet_coords = np.vstack((p_i[legs.FL], p_i[legs.FR], p_i[legs.HR], p_i[legs.HL])).T
    line = plt.plot([], [])[0]
    line.set_data(feet_coords[0], feet_coords[1])
    line.set_3d_properties(feet_coords[2])
    line.set_color("g")
    line.set_marker("o")
    line.set_linestyle("None")

    f_coords = {}
    for leg in legs:
        f_vec = p_i[leg] + f_len * f_i[leg]
        f_coords[leg] = np.vstack((p_i[leg], f_vec)).T
        line = plt.plot([], [])[0]
        line.set_data(f_coords[leg][0], f_coords[leg][1])
        line.set_3d_properties(f_coords[leg][2])
        line.set_color("r")


if __name__ == "__main__":
    from utils import rotMat

    p = np.array([0.0, 0.0, 0.3])
    R = rotMat(np.array([0, 1, 0]), 0.1)
    p_i = {}
    f_i = {}
    for leg in legs:
        p_i[leg] = B_p_Bi[leg]
        f_i[leg] = np.array([0.0, 0.0, 0.2])

    draw(p=p, R=R, p_i=p_i, f_i=f_i)
    plt.show()
