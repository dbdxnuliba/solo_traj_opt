from common import homog, mult_homog_point, legs, B_p_Bi
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.style.use("seaborn")


def draw(p, R, c, f, f_len=0.5):
    T_B = homog(p, R)
    p_Bi = {}
    r_i = {}
    p_i = {}
    for leg in legs:
        p_Bi[leg] = mult_homog_point(T_B, B_p_Bi[leg])
        Bi_p_i = np.array([c[leg][0], 0, c[leg][1]])
        r_i[leg] = B_p_Bi[leg] + Bi_p_i
        p_i[leg] = mult_homog_point(T_B, r_i[leg])

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

    lines = [plt.plot([], [])[0] for _ in range(6)]

    body_coords = np.vstack(
        (p_Bi[legs.FL], p_Bi[legs.FR], p_Bi[legs.HR], p_Bi[legs.HL], p_Bi[legs.FL])).T
    lines[0].set_data(body_coords[0], body_coords[1])
    lines[0].set_3d_properties(body_coords[2])
    lines[0].set_color('b')
    lines[0].set_marker('o')

    feet_coords = np.vstack(
        (p_i[legs.FL], p_i[legs.FR], p_i[legs.HR], p_i[legs.HL])).T
    lines[1].set_data(feet_coords[0], feet_coords[1])
    lines[1].set_3d_properties(feet_coords[2])
    lines[1].set_color('g')
    lines[1].set_marker('o')
    lines[1].set_linestyle('None')

    f_coords = {}
    for leg in legs:
        f_vec = p_i[leg] + f_len * f[leg]
        f_coords[leg] = np.vstack((p_i[leg], f_vec)).T
        lines[2+leg.value].set_data(f_coords[leg][0], f_coords[leg][1])
        lines[2+leg.value].set_3d_properties(f_coords[leg][2])
        lines[2+leg.value].set_color('r')


if __name__ == "__main__":
    from common import rotMat
    p = np.array([0.0, 0.0, 0.3])
    R = rotMat(np.array([0, 1, 0]), 0.1)
    c = {}
    f = {}
    for leg in legs:
        c[leg] = np.array([0.0, -0.2])
        f[leg] = np.array([0.0, 0.0, 0.2])

    draw(p=p, R=R, c=c, f=f)
    plt.show()
