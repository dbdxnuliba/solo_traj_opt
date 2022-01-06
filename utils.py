import numpy as np
import casadi as ca


# given 1x3 vector, returns 3x3 skew symmetric cross product matrix
def skew(s):
    return np.array([[0, -s[2], s[1]], [s[2], 0, -s[0]], [-s[1], s[0], 0]])


# derives a symbolic version of the skew function
def derive_skew():
    s = ca.SX.sym("s", 3)

    skew_sym = ca.SX(3, 3)
    # skew_sym = ca.SX.zeros(3, 3)
    skew_sym[0, 1] = -s[2]
    skew_sym[0, 2] = s[1]
    skew_sym[1, 0] = s[2]
    skew_sym[1, 2] = -s[0]
    skew_sym[2, 0] = -s[1]
    skew_sym[2, 1] = s[0]

    return ca.Function("skew", [s], [skew_sym])


# given axis and angle, returns 3x3 rotation matrix
def rotMat(s, th):
    # normalize s if isn't already normalized
    norm_s = np.linalg.norm(s)
    assert norm_s != 0.0
    s_normalized = s / norm_s

    # Rodrigues' rotation formula
    skew_s = skew(s_normalized)
    return np.eye(3) + np.sin(th) * skew_s + (1.0 - np.cos(th)) * skew_s @ skew_s


# derives a symbolic version of the rotMat function
def derive_rotMat():
    s = ca.SX.sym("s", 3)
    th = ca.SX.sym("th")
    skew_func = derive_skew()
    skew_sym = skew_func(s)

    rotMat_sym = (
        ca.SX.eye(3) + ca.sin(th) * skew_sym + (1 - ca.cos(th)) * skew_sym @ skew_sym
    )
    return ca.Function("rotMat", [s, th], [rotMat_sym])


# given position vector and rotation matrix, returns 4x4 homogeneous
# transformation matrix
def homog(p, R):
    return np.block([[R, p[:, np.newaxis]], [np.zeros((1, 3)), 1]])


# derives a symbolic version of the homog function
def derive_homog():
    p = ca.SX.sym("p", 3)
    R = ca.SX.sym("R", 3, 3)
    homog_sym = ca.SX(4, 4)
    homog_sym[:3, :3] = R
    homog_sym[:3, 3] = p
    homog_sym[3, 3] = 1.0
    return ca.Function("homog", [p, R], [homog_sym])


# derives a symbolic function that reverses the direction of the coordinate
# transformation defined by a 4x4 homogeneous transformation matrix
def derive_reverse_homog():
    T = ca.SX.sym("T", 4, 4)
    R = T[:3, :3]
    p = T[:3, 3]
    reverse_homog_sym = ca.SX(4, 4)
    reverse_homog_sym[:3, :3] = R.T
    reverse_homog_sym[:3, 3] = -R.T @ p
    reverse_homog_sym[3, 3] = 1.0
    return ca.Function("reverse_homog", [T], [reverse_homog_sym])


# multiplication between a 4x4 homogenous transformation matrix and 3x1
# position vector, returns 3x1 position
def mult_homog_point(T, p):
    p_aug = np.concatenate((p, [1.0]))
    return (T @ p_aug)[:3]


# derives a symbolic version of the mult_homog_point function
def derive_mult_homog_point():
    T = ca.SX.sym("T", 4, 4)
    p = ca.SX.sym("p", 3)
    p_aug = ca.SX.ones(4, 1)
    p_aug[:3] = p
    mult_homog_point_sym = (T @ p_aug)[:3]
    return ca.Function("mult_homog_point", [T, p], [mult_homog_point_sym])


# test functions
if __name__ == "__main__":
    x_axis = np.eye(3)[:, 0]
    y_axis = np.eye(3)[:, 1]
    z_axis = np.eye(3)[:, 2]

    print("\ntest skew")
    skew_func = derive_skew()
    print(skew(np.array([1, 2, 3])))
    print(skew_func(np.array([1, 2, 3])))
    s = ca.SX.sym("s", 3)
    print(skew_func(s))

    print("\ntest rotMat")
    rotMat_func = derive_rotMat()
    print(rotMat(x_axis, np.pi / 4))
    print(rotMat_func(x_axis, np.pi / 4))
    print(rotMat(y_axis, np.pi / 4))
    print(rotMat_func(y_axis, np.pi / 4))
    print(rotMat(z_axis, np.pi / 4))
    print(rotMat_func(z_axis, np.pi / 4))
    print(
        np.linalg.norm(
            rotMat(x_axis, np.pi / 4) @ rotMat(x_axis, np.pi / 4).T - np.eye(3)
        )
    )
    print(
        np.linalg.norm(
            rotMat(x_axis, np.pi / 4).T @ rotMat(x_axis, np.pi / 4) - np.eye(3)
        )
    )
    th = ca.SX.sym("th")
    print(rotMat_func(s, th))
    print(
        np.linalg.norm(
            rotMat_func(x_axis, np.pi / 4) @ rotMat_func(x_axis, np.pi / 4).T
            - np.eye(3)
        )
    )

    print("\ntest homog")
    homog_func = derive_homog()
    p = np.array([1, 2, 3])
    R = rotMat(x_axis, np.pi / 4)
    print(homog(p, R))
    print(homog_func(p, R))

    print("\ntest mult_homog_point")
    mult_homog_point_func = derive_mult_homog_point()
    print(mult_homog_point(homog(x_axis, R), y_axis))
    print(mult_homog_point_func(homog(x_axis, R), y_axis))

    reverse_homog_func = derive_reverse_homog()
    T = ca.SX.sym("T", 4, 4)
    print(reverse_homog_func(T))
    T = homog(p, R)
    print(T @ reverse_homog_func(T))
    print(reverse_homog_func(T) @ T)

    import ipdb

    ipdb.set_trace()
