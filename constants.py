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
# g = np.array([0.0, 0.0, -9.81])  # gravity vector
g = 9.81  # gravity constant
mu = 0.7  # friction coefficient

# position of corners of robot, in body frame (so it's a constant)
B_p_Bi = {}
B_p_Bi[legs.FL] = np.array([l_Bx / 2.0, l_By / 2.0, 0.0])
B_p_Bi[legs.FR] = np.array([l_Bx / 2.0, -l_By / 2.0, 0.0])
B_p_Bi[legs.HL] = np.array([-l_Bx / 2.0, l_By / 2.0, 0.0])
B_p_Bi[legs.HR] = np.array([-l_Bx / 2.0, -l_By / 2.0, 0.0])

# global optimization paramters
eps = 1e-6  # numerical zero threshold

# # kinematics constraints paramters
# x_kin_in_lim = l_Bx / 2.0  # half body length to avoid feet collision
# # edge length of largest square that fits within leg workspace
# x_kin_out_lim = (l_thigh + l_calf) / np.sqrt(2)
# z_kin_lower_lim = -(l_thigh + l_calf) / np.sqrt(2)
# z_kin_upper_lim = (l_thigh + l_calf) / np.sqrt(2)

# # LQR weights
# Q_p = np.array([1000.0, 1000.0, 1000.0])
# Q_p_i = np.array([100.0, 100.0, 100.0])
# Q_R = np.array([100.0, 100.0, 100.0])
# Q_pdot = np.array([10.0, 10.0, 10.0])
# Q_omega = np.array([1.0, 1.0, 1.0])
# Q_f_i = np.array([0.1, 0.1, 0.1])
# R_p_i_dot = np.array([1.0, 1.0, 1.0])

# # matrix used for rotation matrix cost, calculated from above values
# Kp_vec = np.linalg.solve(
#     np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]), 4.0 * Q_R
# )  # 3 element vector
# Gp = sum(Kp_vec) * np.eye(3) - np.diag(Kp_vec)  # 3x3 matrix

# BCD objective and constraint parameters
phi_r = 1000.0  # COM reference tracking
phi_l = 10.0  # momentum reference tracking
phi_k = 100.0  # angular momentum reference tracking
L_r = phi_r / 10.0  # COM previous solution regularization
L_l = phi_l  # momentum previous solution regularization
L_k = phi_k  # angular momentum previous solution regularization
L_p = 100.0  # foot previous solution regularization
psi_f = 0.1  # force regularization
L_kin = l_thigh + l_calf + l_Bx / 2  # max dist from COM to feet
eps_contact = 1e-3  # ground contact distance threshold

# optimization problem dimensionality parameters, per timestep
dim_x = 33  # decision variables for global optimization
dim_xR = dim_x + 9  # decision variables + base rotation matrix
dim_x_fqp = 21  # decision variables for force QP
dim_dyn_fqp = 9  # dynamic constraints for force QP
dim_fric_fqp = 20  # friction constraints for force QP
dim_kin_fqp = 32  # kinematic constraints for force QP
dim_x_cqp = 18  # decision variables for contact QP
dim_dyn_cqp = 6  # dynamic constraints for contact QP
dim_loc_cqp = 12  # location constraints for contact QP
dim_kin_cqp = 32  # kinematic constraints for contact QP

# QP solver settings
osqp_settings = {}
osqp_settings["verbose"] = False
osqp_settings["eps_abs"] = 1e-7
osqp_settings["eps_rel"] = 1e-7
osqp_settings["eps_prim_inf"] = 1e-6
osqp_settings["eps_dual_inf"] = 1e-6
osqp_settings["polish"] = True
osqp_settings["scaled_termination"] = True
osqp_settings["adaptive_rho"] = True
osqp_settings["check_termination"] = 50
