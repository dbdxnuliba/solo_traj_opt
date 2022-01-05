import numpy as np
from numpy import sin, cos
import enum

# HELPER FUNCTIONS


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


# PARAMETERS/CONSTANTS

# parameters
l_Bx = 0.356
l_By = 0.2
l_thigh = 0.16
l_calf = 0.16


# enum for the four legs
class legs(enum.Enum):
    FL = 0
    FR = 1
    HL = 2
    HR = 3


# position of corners of robot, in body frame (so it's a constant)
B_p_Bi = {}
B_p_Bi[legs.FL] = np.array([l_Bx / 2.0, l_By / 2.0, 0.0])
B_p_Bi[legs.FR] = np.array([l_Bx / 2.0, -l_By / 2.0, 0.0])
B_p_Bi[legs.HL] = np.array([-l_Bx / 2.0, l_By / 2.0, 0.0])
B_p_Bi[legs.HR] = np.array([-l_Bx / 2.0, -l_By / 2.0, 0.0])
