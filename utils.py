import enum
import numpy as np
import casadi as ca

# enum for the four legs
class legs(enum.Enum):
    FL = 0
    FR = 1
    HL = 2
    HR = 3


# given 1x3 vector, returns 3x3 skew symmetric cross product matrix
def skew_np(s):
    return np.array([[0, -s[2], s[1]], [s[2], 0, -s[0]], [-s[1], s[0], 0]])


# derives a symbolic version of the skew function
def derive_skew_ca():
    s = ca.SX.sym("s", 3)

    skew_sym = ca.SX(3, 3)
    # skew_sym = ca.SX.zeros(3, 3)
    skew_sym[0, 1] = -s[2]
    skew_sym[0, 2] = s[1]
    skew_sym[1, 0] = s[2]
    skew_sym[1, 2] = -s[0]
    skew_sym[2, 0] = -s[1]
    skew_sym[2, 1] = s[0]

    return ca.Function("skew_ca", [s], [skew_sym])


# given axis and angle, returns 3x3 rotation matrix
def rot_mat_np(s, th):
    # normalize s if isn't already normalized
    norm_s = np.linalg.norm(s)
    assert norm_s != 0.0
    s_normalized = s / norm_s

    # Rodrigues' rotation formula
    skew_s = skew_np(s_normalized)
    return np.eye(3) + np.sin(th) * skew_s + (1.0 - np.cos(th)) * skew_s @ skew_s


# derives a symbolic version of the rotMat function
def derive_rot_mat_ca():
    s = ca.SX.sym("s", 3)
    th = ca.SX.sym("th")
    skew_ca = derive_skew_ca()
    skew_sym = skew_ca(s)

    rot_mat_sym = (
        ca.SX.eye(3) + ca.sin(th) * skew_sym + (1 - ca.cos(th)) * skew_sym @ skew_sym
    )
    return ca.Function("rot_mat_ca", [s, th], [rot_mat_sym])


# given position vector and rotation matrix, returns 4x4 homogeneous
# transformation matrix
def homog_np(p, R):
    return np.block([[R, p[:, np.newaxis]], [np.zeros((1, 3)), 1]])


# derives a symbolic version of the homog function
def derive_homog_ca():
    p = ca.SX.sym("p", 3)
    R = ca.SX.sym("R", 3, 3)
    homog_sym = ca.SX(4, 4)
    homog_sym[:3, :3] = R
    homog_sym[:3, 3] = p
    homog_sym[3, 3] = 1.0
    return ca.Function("homog_ca", [p, R], [homog_sym])


# derives a symbolic function that reverses the direction of the coordinate
# transformation defined by a 4x4 homogeneous transformation matrix
def derive_reverse_homog_ca():
    T = ca.SX.sym("T", 4, 4)
    R = T[:3, :3]
    p = T[:3, 3]
    reverse_homog_sym = ca.SX(4, 4)
    reverse_homog_sym[:3, :3] = R.T
    reverse_homog_sym[:3, 3] = -R.T @ p
    reverse_homog_sym[3, 3] = 1.0
    return ca.Function("reverse_homog_ca", [T], [reverse_homog_sym])


# multiplication between a 4x4 homogenous transformation matrix and 3x1
# position vector, returns 3x1 position
def mult_homog_point_np(T, p):
    p_aug = np.concatenate((p, [1.0]))
    return (T @ p_aug)[:3]


# derives a symbolic version of the mult_homog_point function
def derive_mult_homog_point_ca():
    T = ca.SX.sym("T", 4, 4)
    p = ca.SX.sym("p", 3)
    p_aug = ca.SX.ones(4, 1)
    p_aug[:3] = p
    mult_homog_point_sym = (T @ p_aug)[:3]
    return ca.Function("mult_homog_point_ca", [T, p], [mult_homog_point_sym])


def extract_state_ca(X, U, k):
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


def flatten_state_ca(p, R, pdot, omega, p_i, f_i):
    R_flat = ca.reshape(R, 9, 1)
    p_i_flat = ca.MX(12, 1)
    f_i_flat = ca.MX(12, 1)
    for leg in legs:
        p_i_flat[3 * leg.value : leg.value * 3 + 3] = p_i[leg]
        f_i_flat[3 * leg.value : leg.value * 3 + 3] = f_i[leg]

    X_k = ca.vertcat(p, R_flat, pdot, omega)
    U_k = ca.vertcat(p_i_flat, f_i_flat)

    return X_k, U_k


# test functions
if __name__ == "__main__":
    x_axis = np.eye(3)[:, 0]
    y_axis = np.eye(3)[:, 1]
    z_axis = np.eye(3)[:, 2]

    print("\ntest skew")
    skew_ca = derive_skew_ca()
    print(skew_np(np.array([1, 2, 3])))
    print(skew_ca(np.array([1, 2, 3])))
    s = ca.SX.sym("s", 3)
    print(skew_ca(s))

    print("\ntest rotMat")
    rot_mat_ca = derive_rot_mat_ca()
    print(rot_mat_np(x_axis, np.pi / 4))
    print(rot_mat_ca(x_axis, np.pi / 4))
    print(rot_mat_np(y_axis, np.pi / 4))
    print(rot_mat_ca(y_axis, np.pi / 4))
    print(rot_mat_np(z_axis, np.pi / 4))
    print(rot_mat_ca(z_axis, np.pi / 4))
    print(
        np.linalg.norm(
            rot_mat_np(x_axis, np.pi / 4) @ rot_mat_np(x_axis, np.pi / 4).T - np.eye(3)
        )
    )
    print(
        np.linalg.norm(
            rot_mat_np(x_axis, np.pi / 4).T @ rot_mat_np(x_axis, np.pi / 4) - np.eye(3)
        )
    )
    th = ca.SX.sym("th")
    print(rot_mat_ca(s, th))
    print(
        np.linalg.norm(
            rot_mat_ca(x_axis, np.pi / 4) @ rot_mat_ca(x_axis, np.pi / 4).T - np.eye(3)
        )
    )

    print("\ntest homog")
    homog_ca = derive_homog_ca()
    p = np.array([1, 2, 3])
    R = rot_mat_np(x_axis, np.pi / 4)
    print(homog_np(p, R))
    print(homog_ca(p, R))

    print("\ntest mult_homog_point")
    mult_homog_point_ca = derive_mult_homog_point_ca()
    print(mult_homog_point_np(homog_np(x_axis, R), y_axis))
    print(mult_homog_point_ca(homog_np(x_axis, R), y_axis))

    reverse_homog_ca = derive_reverse_homog_ca()
    T = ca.SX.sym("T", 4, 4)
    print(reverse_homog_ca(T))
    T = homog_np(p, R)
    print(T @ reverse_homog_ca(T))
    print(reverse_homog_ca(T) @ T)

    print("\ntest extract_state")
    X = ca.SX.sym("X", 18, 3)
    U = ca.SX.sym("U", 24, 3)
    p, R, pdot, omega, p_i, f_i = extract_state_ca(X, U, 0)
    print("p:", p)
    print("R:", R)
    print("pdot:", pdot)
    print("omega:", omega)
    for leg in legs:
        print("p_i[", leg.value, "]:", p_i[leg])
        print("f_i[", leg.value, "]:", f_i[leg])

    import ipdb

    ipdb.set_trace()
