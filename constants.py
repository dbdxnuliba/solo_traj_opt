import enum
import numpy as np

# enum for the four legs
class legs(enum.Enum):
    FL = 0
    FR = 1
    HL = 2
    HR = 3


# robot physical length paramters
l_Bx = 0.380  # length of body, measured axis to axis at hip (from CAD)
# width between left and right feet (from CAD)
# outer face to face was 310mm, inner face to face was 290mm, thickness of lower leg is 10mm
l_By = 0.3
l_thigh = 0.165  # length of upper leg module measured axis to axis (from CAD)
l_calf = 0.160  # length of lower leg measured axis to axis (from CAD)

# robot inertial paramters
# TODO: check if mass of motors is included and below inertial properties
m = 1.43315091  # mass of body  (from URDF)
# body moment of inertia in body frame (from URDF)
B_I = np.diag([0.00578574, 0.01938108, 0.02476124])
B_I_inv = np.diag(1 / np.array([0.00578574, 0.01938108, 0.02476124]))

# physical parameters external to robot
g = np.array([0.0, 0.0, -9.81])  # gravity vector
mu = 0.7  # friction coefficient

# position of corners of robot, in body frame (so it's a constant)
B_p_Bi = {}
B_p_Bi[legs.FL] = np.array([l_Bx / 2.0, l_By / 2.0, 0.0])
B_p_Bi[legs.FR] = np.array([l_Bx / 2.0, -l_By / 2.0, 0.0])
B_p_Bi[legs.HL] = np.array([-l_Bx / 2.0, l_By / 2.0, 0.0])
B_p_Bi[legs.HR] = np.array([-l_Bx / 2.0, -l_By / 2.0, 0.0])

# global optimization paramters
eps = 1e-6  # numerical zero threshold

# kinematics constraints paramters
x_kin_in_lim = l_Bx / 2.0  # half body length to avoid feet collision
# edge length of largest square that fits within leg workspace
x_kin_out_lim = (l_thigh + l_calf) / np.sqrt(2)
z_kin_lower_lim = -(l_thigh + l_calf) / np.sqrt(2)
z_kin_upper_lim = (l_thigh + l_calf) / np.sqrt(2)

# LQR weights
Q_p = np.array([100.0, 100.0, 100.0])
Q_p_i = np.array([100.0, 100.0, 100.0])
Q_R = np.array([100.0, 100.0, 100.0])
Q_pdot = np.array([10.0, 10.0, 10.0])
Q_omega = np.array([1.0, 1.0, 1.0])
Q_f_i = np.array([0.1, 0.1, 0.1])

# matrix used for rotation matrix cost, calculated from above values
Kp_vec = np.linalg.solve(
    np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]), 2.0 * Q_R
)  # 3 element vector
Gp = sum(Kp_vec) * np.eye(3) - np.diag(Kp_vec)  # 3x3 matrix
