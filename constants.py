import numpy as np
import enum


l_Bx = 0.380  # length of body, measured axis to axis at hip (from CAD)
# width between left and right feet (from CAD)
# outer face to face was 310mm, inner face to face was 290mm, thickness of lower leg is 10mm
l_By = 0.3
l_thigh = 0.165  # length of upper leg module measured axis to axis (from CAD)
l_calf = 0.160  # length of lower leg measured axis to axis (from CAD)

# TODO: check if mass of motors is included and below inertial properties
m = 1.43315091  # mass of body  (from URDF)
# body moment of inertia in body frame (from URDF)
B_I = np.diag([0.00578574, 0.01938108, 0.02476124])


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
