import numpy as np
from numpy import sin, cos


# given position vector and rotation matrix, returns 4x4 homogeneous
# transformation matrix
def homog(p, R):
    return np.block([[R, p[:, np.newaxis]], [np.zeros((1, 3)), 1]])


# multiplication between a 4x4 homogenous transformation matrix and 3x1
# position vector, returns 3x1 position
def mult_homog_point(T, p):
    p_aug = np.concatenate((p, [1.0]))
    return (T @ p_aug)[:3]


# given 1x3 vector, returns 3x3 skew symmetric cross product matrix
def skew(s):
    return np.array([[0, -s[2], s[1]], [s[2], 0, -s[0]], [-s[1], s[0], 0]])


# given axis and angle, returns 3x3 rotation matrix
def rotMat(s, th):
    # normalize s if isn't already normalized
    norm_s = np.linalg.norm(s)
    assert norm_s != 0.0
    s_normalized = s / norm_s

    # Rodrigues' rotation formula
    skew_s = skew(s_normalized)
    return np.eye(3) + sin(th) * skew_s + (1.0 - cos(th)) * skew_s @ skew_s
