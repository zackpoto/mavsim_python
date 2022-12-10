"""
compute_ss_model
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        2/4/2019 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
from scipy.optimize import minimize
from tools.rotations import Euler2Quaternion, Quaternion2Euler
import parameters.aerosonde_parameters as MAV
from parameters.simulation_parameters import ts_simulation as Ts
from message_types.msg_delta import MsgDelta


def compute_model(mav, trim_state, trim_input):
    A_lon, B_lon, A_lat, B_lat = compute_ss_model(mav, trim_state, trim_input)
    Va_trim, alpha_trim, theta_trim, a_phi1, a_phi2, a_theta1, a_theta2, a_theta3, \
    a_V1, a_V2, a_V3 = compute_tf_model(mav, trim_state, trim_input)

    # write transfer function gains to file
    file = open('model_coef.py', 'w')
    file.write('import numpy as np\n')
    file.write('x_trim = np.array([[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f]]).T\n' %
               (trim_state.item(0), trim_state.item(1), trim_state.item(2), trim_state.item(3),
                trim_state.item(4), trim_state.item(5), trim_state.item(6), trim_state.item(7),
                trim_state.item(8), trim_state.item(9), trim_state.item(10), trim_state.item(11),
                trim_state.item(12)))
    file.write('u_trim = np.array([[%f, %f, %f, %f]]).T\n' %
               (trim_input.elevator, trim_input.aileron, trim_input.rudder, trim_input.throttle))
    file.write('Va_trim = %f\n' % Va_trim)
    file.write('alpha_trim = %f\n' % alpha_trim)
    file.write('theta_trim = %f\n' % theta_trim)
    file.write('a_phi1 = %f\n' % a_phi1)
    file.write('a_phi2 = %f\n' % a_phi2)
    file.write('a_theta1 = %f\n' % a_theta1)
    file.write('a_theta2 = %f\n' % a_theta2)
    file.write('a_theta3 = %f\n' % a_theta3)
    file.write('a_V1 = %f\n' % a_V1)
    file.write('a_V2 = %f\n' % a_V2)
    file.write('a_V3 = %f\n' % a_V3)
    file.write('A_lon = np.array([\n    [%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f]])\n' %
    (A_lon[0][0], A_lon[0][1], A_lon[0][2], A_lon[0][3], A_lon[0][4],
     A_lon[1][0], A_lon[1][1], A_lon[1][2], A_lon[1][3], A_lon[1][4],
     A_lon[2][0], A_lon[2][1], A_lon[2][2], A_lon[2][3], A_lon[2][4],
     A_lon[3][0], A_lon[3][1], A_lon[3][2], A_lon[3][3], A_lon[3][4],
     A_lon[4][0], A_lon[4][1], A_lon[4][2], A_lon[4][3], A_lon[4][4]))
    file.write('B_lon = np.array([\n    [%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f]])\n' %
    (B_lon[0][0], B_lon[0][1],
     B_lon[1][0], B_lon[1][1],
     B_lon[2][0], B_lon[2][1],
     B_lon[3][0], B_lon[3][1],
     B_lon[4][0], B_lon[4][1],))
    file.write('A_lat = np.array([\n    [%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f]])\n' %
    (A_lat[0][0], A_lat[0][1], A_lat[0][2], A_lat[0][3], A_lat[0][4],
     A_lat[1][0], A_lat[1][1], A_lat[1][2], A_lat[1][3], A_lat[1][4],
     A_lat[2][0], A_lat[2][1], A_lat[2][2], A_lat[2][3], A_lat[2][4],
     A_lat[3][0], A_lat[3][1], A_lat[3][2], A_lat[3][3], A_lat[3][4],
     A_lat[4][0], A_lat[4][1], A_lat[4][2], A_lat[4][3], A_lat[4][4]))
    file.write('B_lat = np.array([\n    [%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f]])\n' %
    (B_lat[0][0], B_lat[0][1],
     B_lat[1][0], B_lat[1][1],
     B_lat[2][0], B_lat[2][1],
     B_lat[3][0], B_lat[3][1],
     B_lat[4][0], B_lat[4][1],))
    file.write('Ts = %f\n' % Ts)
    file.close()


def compute_tf_model(mav, trim_state, trim_input):
    # trim values
    mav._state = trim_state
    mav._update_velocity_data()
    Va_trim = mav._Va
    alpha_trim = mav._alpha
    phi, theta_trim, psi = Quaternion2Euler(trim_state[6:10])

    # define transfer function constants
    #CHAP5 - slide 21
    a_phi1 = -(MAV.rho*Va_trim**2*MAV.S_wing*MAV.b/2) * MAV.C_p_p * MAV.b/(2*Va_trim)
    a_phi2 = (MAV.rho*Va_trim**2*MAV.S_wing*MAV.b/2) * MAV.C_p_delta_a
    #CHAP5 - slide 28
    a_theta1 = -(MAV.rho*Va_trim**2*MAV.c*MAV.S_wing/(2*MAV.Jy)) * MAV.C_m_q * MAV.c / (2*Va_trim)
    a_theta2 = -(MAV.rho*Va_trim**2*MAV.c*MAV.S_wing/(2*MAV.Jy)) * MAV.C_m_alpha
    a_theta3 = (MAV.rho*Va_trim**2*MAV.c*MAV.S_wing/(2*MAV.Jy)) * MAV.C_m_delta_e

    # Compute transfer function coefficients using new propulsion model
    # CHAP5 - slide 34
    a_V1 = MAV.rho*Va_trim*MAV.S_wing/MAV.mass * (MAV.C_D_0 + MAV.C_D_alpha*alpha_trim + MAV.C_D_delta_e*trim_input.elevator) \
        - (1/MAV.mass) * dT_dVa(mav, Va_trim, trim_input.throttle)
    a_V2 = (1/MAV.mass) * dT_ddelta_t(mav, Va_trim, trim_input.throttle)
    a_V3 = MAV.gravity*np.cos(theta_trim-alpha_trim)

    return Va_trim, alpha_trim, theta_trim, a_phi1, a_phi2, a_theta1, a_theta2, a_theta3, a_V1, a_V2, a_V3


def compute_ss_model(mav, trim_state, trim_input):
    x_euler = euler_state(trim_state)
    A = df_dx(mav, x_euler, trim_input)
    B = df_du(mav, x_euler, trim_input)

    # extract longitudinal states (u, w, q, theta, pd) and change pd to h
    #CHAP5 - slide 51

    E1 = np.array([
        [0.0, 0.0,  0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0,  0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])

    E2 = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ])

    A_lon = E1 @ A @ E1.T
    B_lon = E1 @ B @ E2.T

    # extract lateral states (v, p, r, phi, psi)
    #CHAP5 - slide 52

    E3 = np.array([
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    ])

    E4 = np.array([
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    A_lat = E3 @ A @ E3.T
    B_lat = E3 @ B @ E4.T
    return A_lon, B_lon, A_lat, B_lat

def euler_state(x_quat):
    # convert state x with attitude represented by quaternion
    # to x_euler with attitude represented by Euler angles

    #CHAP5 - slide 48
    p = x_quat[0:3]
    v = x_quat[3:6]
    q = x_quat[6:10]
    theta = np.array([Quaternion2Euler(q)]).T
    w = x_quat[10:13]

    x_euler = np.concatenate((p, v, theta, w), axis=0)
    return x_euler

def quaternion_state(x_euler):
    # convert state x_euler with attitude represented by Euler angles
    # to x_quat with attitude represented by quaternions

    #CHAP5 - slide 48
    p = x_euler[0:3]
    v = x_euler[3:6]
    theta = x_euler[6:9]
    q = Euler2Quaternion(theta.item(0), theta.item(1), theta.item(2))
    w = x_euler[9:12]

    x_quat = np.concatenate((p, v, q, w), axis=0)
    return x_quat

def f_euler(mav, x_euler, delta):
    # return 12x1 dynamics (as if state were Euler state)
    # compute f at euler_state
    forces_moments = mav._forces_moments(delta)

    x_quat = quaternion_state(x_euler)
    f_quat = mav._derivatives(x_quat, forces_moments)
    f_euler_ = euler_state(f_quat)
    return f_euler_

def df_dx(mav, x_euler, delta):
    # CHAP5 - slide 47

    # take partial of f_euler with respect to x_euler
    m = 12
    n = 12
    eps = 0.001
    A = np.zeros((m, n))
    f_at_x = f_euler(mav, x_euler, delta)
    for i in range(0, n):
        x_eps = np.copy(x_euler)
        x_eps[i][0] += eps
        f_at_x_eps = f_euler(mav, x_eps, delta)
        df_dxi = (f_at_x_eps-f_at_x)/eps
        A[:, i] = df_dxi[:, 0]
    return A


def df_du(mav, x_euler, delta):
    # CHAP5 - slide 47 (modified for u)
    # take partial of f_euler with respect to input
    m = 12
    n = 4
    eps = 0.001
    B = np.zeros((m, n))
    f_at_u = f_euler(mav, x_euler, delta)
    for i in range(0, n):
        u_eps = delta.to_array()
        u_eps[i] += eps
        delta_eps = MsgDelta()
        delta_eps.from_array(u_eps)
        f_at_u_eps = f_euler(mav, x_euler, delta_eps)
        df_dui = (f_at_u_eps-f_at_u)/eps
        B[:, i] = df_dui[:, 0]
    return B


def dT_dVa(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to Va
    eps = 0.001
    T_eps, Q_eps = mav._motor_thrust_torque(Va + eps, delta_t)
    T, Q = mav._motor_thrust_torque(Va, delta_t)
    return (T_eps - T) / eps

def dT_ddelta_t(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to delta_t
    print("SOMETHING", delta_t)
    eps = 0.001
    T_eps, Q_eps = mav._motor_thrust_torque(Va, delta_t + eps)
    T, Q = mav._motor_thrust_torque(Va, delta_t)
    print((T_eps - T)/eps/11)
    return (T_eps - T) / eps