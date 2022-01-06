import numpy as np
import enum


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
