from constants import *
import numpy as np
import scipy.sparse as sp
import casadi as ca


# given 1x3 vector, returns 3x3 skew symmetric cross product matrix
def skew_np(s):
    return np.array([[0, -s[2], s[1]], [s[2], 0, -s[0]], [-s[1], s[0], 0]])


# sparse skew matrix
def skew_sp(s):
    skew_sp = sp.lil_matrix((3, 3))
    skew_sp[0, 1] = -s[2]
    skew_sp[0, 2] = s[1]
    skew_sp[1, 0] = s[2]
    skew_sp[1, 2] = -s[0]
    skew_sp[2, 0] = -s[1]
    skew_sp[2, 1] = s[0]
    return skew_sp


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


# 2D rotation matrix
def rot_mat_2d_np(th):
    return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])


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


# reverses the direction of the coordinate transformation defined by a 4x4
# homogeneous transformation matrix
def reverse_homog_np(T):
    R = T[:3, :3]
    p = T[:3, 3]
    reverse_homog = np.zeros((4, 4))
    reverse_homog[:3, :3] = R.T
    reverse_homog[:3, 3] = -R.T @ p
    reverse_homog[3, 3] = 1.0
    return reverse_homog


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


# generic planar 2 link inverse kinematics implementation
# returns the closest point within the workspace if the requested point is
# outside of it
def planar_IK_np(l1, l2, x, y, elbow_up):
    l = np.sqrt(x**2.0 + y**2.0)
    l = max(abs(l1 - l2), min(l, l1 + l2))

    alpha = np.arctan2(y, x)

    cos_beta = (l**2 + l1**2 - l2**2.0) / (2.0 * l * l1)
    cos_beta = max(-1.0, min(cos_beta, 1.0))
    beta = np.arccos(cos_beta)

    cos_th2_abs = (l**2 - l1**2.0 - l2**2.0) / (2.0 * l1 * l2)
    cos_th2_abs = max(-1.0, min(cos_th2_abs, 1.0))
    th2_abs = np.arccos(cos_th2_abs)

    if elbow_up:
        th1 = alpha - beta
        th2 = th2_abs
    else:
        th1 = alpha + beta
        th2 = -th2_abs

    return th1, th2


# Solo specific functions below

# position of corners of robot, in body frame (so it's a constant)
B_T_Bi = {}
for leg in legs:
    B_T_Bi[leg] = homog_np(B_p_Bi[leg], np.eye(3))


# given numpy trajectory matrix, extract state and rotation R at timestep k
# note the order argument in reshape, which is necessary to make it consistent
# with casadi's reshape
def extract_state_and_rot_np(XR, t):
    r = XR[:3, t]
    l = XR[3:6, t]
    k = XR[6:9, t]
    p_i = {}
    f_i = {}
    for leg in legs:
        p_i[leg] = XR[9 + 3 * leg.value : 9 + leg.value * 3 + 3, t]
        f_i[leg] = XR[21 + 3 * leg.value : 21 + leg.value * 3 + 3, t]
    R_flat = XR[-9:, t]
    R = np.reshape(R_flat, (3, 3), order="F")
    return r, l, k, p_i, f_i, R


# given numpy trajectory matrix, extract state at timestep k
# note the order argument in reshape, which is necessary to make it consistent
# with casadi's reshape
def extract_state_np(X, t):
    r = X[:3, t]
    l = X[3:6, t]
    k = X[6:9, t]
    p_i = {}
    f_i = {}
    for leg in legs:
        p_i[leg] = X[9 + 3 * leg.value : 9 + leg.value * 3 + 3, t]
        f_i[leg] = X[21 + 3 * leg.value : 21 + leg.value * 3 + 3, t]
    return r, l, k, p_i, f_i


# # given casadi trajectory matrix, extract state at timestep k
# def extract_state_ca(X, U, k):
#     p = X[:3, k]
#     R_flat = X[3:12, k]
#     R = ca.reshape(R_flat, 3, 3)
#     pdot = X[12:15, k]
#     omega = X[15:18, k]
#     p_i = {}
#     f_i = {}
#     for leg in legs:
#         p_i[leg] = U[3 * leg.value : leg.value * 3 + 3, k]
#         f_i[leg] = U[12 + 3 * leg.value : 12 + leg.value * 3 + 3, k]
#     return p, R, pdot, omega, p_i, f_i


# given a numpy state and rotation R, flattens it into the same form as a column of a
# trajectory matrix
def flatten_state_and_rot_np(r, l, k, p_i, f_i, R):
    p_i_flat = np.zeros(12)
    f_i_flat = np.zeros(12)
    for leg in legs:
        p_i_flat[3 * leg.value : leg.value * 3 + 3] = p_i[leg]
        f_i_flat[3 * leg.value : leg.value * 3 + 3] = f_i[leg]
    R_flat = np.reshape(R, 9, order="F")

    XR_t = np.hstack((r, l, k, p_i_flat, f_i_flat, R_flat))

    return XR_t


# extract state trajectory from state + rotation matrix trajectory
def extract_X_from_XR(XR):
    X = XR[:-9, :]
    assert X.shape[0] == dim_x
    return X


# inverse kinematics for the solo 8 robot
def solo_IK_np(p, R, p_i):
    T_B = homog_np(p, R)
    q_i = {}
    for leg in legs:
        T_Bi = T_B @ B_T_Bi[leg]
        Bi_T = reverse_homog_np(T_Bi)
        Bi_p_i = mult_homog_point_np(Bi_T, p_i[leg])
        rotate_90 = rot_mat_2d_np(np.pi / 2.0)
        x_z = rotate_90 @ np.array([Bi_p_i[0], Bi_p_i[2]])
        if leg == legs.FL or leg == legs.FR:
            q1, q2 = planar_IK_np(l_thigh, l_calf, x_z[0], x_z[1], True)
        else:
            q1, q2 = planar_IK_np(l_thigh, l_calf, x_z[0], x_z[1], False)
        q_i[leg] = np.array([q1, q2])

    return q_i


# test functions
if __name__ == "__main__":
    print("\ntest flatten_state_np")
    r = np.array([1.0, 2.0, 3.0])
    l = np.array([0.4, 0.5, 0.6])
    k = np.array([3, 4, 5])
    p_i = {}
    f_i = {}
    for leg in legs:
        p_i[leg] = leg.value + np.array([0.7, 0.8, 0.9])
        f_i[leg] = leg.value + np.array([0.07, 0.08, 0.09])
    R = rot_mat_np(np.array([0, 1, 0]), np.pi / 4.0)

    XR_t = flatten_state_and_rot_np(r, l, k, p_i, f_i, R)
    print(XR_t)

    print("\ntest extract_state_np")
    (
        r_extracted,
        l_extracted,
        k_extracted,
        p_i_extracted,
        f_i_extracted,
        R_extracted,
    ) = extract_state_and_rot_np(XR_t[:, np.newaxis], 0)
    print("r_extracted", r_extracted)
    print("l_extracted", l_extracted)
    print("k_extracted", k_extracted)
    print("p_i_extracted", p_i_extracted)
    print("f_i_extracted", f_i_extracted)
    print("R_extracted", R_extracted)

    print("skew_np(r)", skew_np(r))
    print("skew_sp(r).toarray()", skew_sp(r).toarray())

    import ipdb

    ipdb.set_trace()
