"""
mavsimPy
    - Chapter 3 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        12/18/2018 - RWB
        1/14/2019 - RWB
"""
import sys
import os


sys.path.append(os.path.split(sys.path[0])[0])

import numpy as np
import parameters.simulation_parameters as SIM

import parameters.aerosonde_parameters as MAV
from tools.rotations import Euler2Quaternion
from tools.rotations import Euler2Rotation
from tools.rotations import Rotation2Quaternion
from tools.rotations import Quaternion2Euler

from chap2.mav_viewer import MavViewer
from chap3.data_viewer import DataViewer
from chap3.mav_dynamics import MavDynamics
from message_types.msg_delta import MsgDelta

# initialize the visualization
VIDEO = False  # True==write video, False==don't write video
ANIMATE = True
if ANIMATE:
    mav_view = MavViewer()  # initialize the mav viewer
    data_view = DataViewer()  # initialize view of data plots
if VIDEO and ANIMATE:
    from chap2.video_writer import VideoWriter
    video = VideoWriter(video_name="projectile_test.mp4",
                        bounding_box=(0, 0, 1000, 1000),
                        output_rate=SIM.ts_video)

# initialize elements of the architecture
mav = MavDynamics(SIM.ts_simulation)
delta = MsgDelta()

# initialize the simulation time
sim_time = SIM.start_time

# define initial conditions for projectile
north0 = 0.
east0 = 0.
altitude0 = 100
u0 = 0
phi0 = np.pi * (0)  # initial roll angle
theta0 = np.pi * (-1/2)  # initial pitch angle
psi0 = np.pi * (0)  # initial yaw angle
e = Euler2Quaternion(phi0, theta0, psi0)
e0 = e.item(0)
e1 = e.item(1)
e2 = e.item(2)
e3 = e.item(3)
mav._state = np.array([
    [north0],         # (north0)
    [east0],         # (east0)
    [-altitude0],         # (down0)
    [u0],    # (u0)
    [0.],    # (v0)
    [0.],    # (w0)
    [e0],    # (e0)
    [e1],    # (e1)
    [e2],    # (e2)
    [e3],    # (e3)
    [0.],    # (p0)
    [0.],    # (q0)
    [0.],    # (r0)
])
mav._update_true_state()

V0_z = u0 * np.sin(theta0)
V0_x = np.cos(psi0) * u0 * np.cos(theta0)
V0_y = np.sin(psi0) * u0 * np.cos(theta0)

print(V0_x, V0_y, V0_z)

t_flight_expected = (V0_z + np.sqrt(V0_z**2 + 2*MAV.gravity*altitude0))/MAV.gravity
n_land = north0 + t_flight_expected * V0_x
e_land = east0 + t_flight_expected * V0_y
if V0_z > 0:
    max_altitude = altitude0 + V0_z**2 / (2 * MAV.gravity)
    V_impact_z = np.sqrt(2*MAV.gravity*max_altitude)
else:
    max_altitude = altitude0
    V_impact_z = V0_z + np.sqrt(2*MAV.gravity*max_altitude)

V_impact = np.sqrt(V_impact_z**2 + V0_y**2 + V0_x**2)

last_max_alt = altitude0
sim_time = 0
assert altitude0 >= 0
while sim_time < t_flight_expected + 100:

    # this keeps gravity in z-direction of inertial frame
    Rb_i = Euler2Rotation(mav.true_state.phi, mav.true_state.theta, mav.true_state.psi).T
    f_i = np.array([0, 0, MAV.gravity*MAV.mass]).T
    f_b = Rb_i @ f_i
    forces_moments = np.array([[f_b[0], f_b[1], f_b[2], 0, 0, 0]]).T

    mav.update(forces_moments)  # propagate the MAV dynamics

    if ANIMATE:
        data_view.update(mav.true_state,  # true states
                        mav.true_state,  # estimated states
                        mav.true_state,  # commanded states
                        delta,  # inputs to the aircraft
                        SIM.ts_simulation)

        mav_view.update(mav.true_state)  # plot body of MAV

    if VIDEO and ANIMATE:
        video.update(sim_time)          # record frame

    if last_max_alt < mav.true_state.altitude:
        last_max_alt = mav.true_state.altitude
    sim_time += SIM.ts_simulation

    # LANDING + COMPARE SIM AND PHYSICS VALUES
    if mav.true_state.altitude <= 0:
        ## FLIGHT TIME UNIT TEST
        if (sim_time - t_flight_expected) < SIM.ts_simulation:
            print("\nPASSED!!")
        else:
            print("FAILED!")
        print("Time to fall from %d is %f and expected to be %f \n" %(altitude0, sim_time, t_flight_expected))

        ## LANDING POSITION (NORTH) UNIT TEST
        n_pos_error = SIM.ts_simulation * max(np.abs(V0_x), 1)
        if np.abs(mav.true_state.north - n_land) < n_pos_error:
            print("PASSED!!")
        else:
            print("FAILED!")
        print("Expect to land at %f N and it was %f N \n" %(n_land, mav.true_state.north))

        ## LANDING POSITION (EAST) UNIT TEST
        e_pos_error = SIM.ts_simulation * max(np.abs(V0_y), 1)
        if np.abs(mav.true_state.east - e_land) < e_pos_error:
            print("PASSED!!")
        else:
            print("FAILED!")
        print("Expect to land at %f E and it was %f E \n" %(e_land, mav.true_state.east))

        ## MAX ALTITUDE UNIT TEST
        alt_error = SIM.ts_simulation * max(np.abs(V0_z), 1)
        if np.abs(max_altitude - last_max_alt) < alt_error:
            print("PASSED!!")
        else:
            print("FAILED!")
        print("Expected max height to be %f and it was %f \n" %(max_altitude, last_max_alt))

        ## IMPACT VELOCITY
        Vf_error = SIM.ts_simulation * max(np.abs(V0_x), np.abs(V0_y), np.abs(V0_z), 1)
        V = np.linalg.norm((mav._state[3:6]))
        if np.abs(V_impact - V) < Vf_error:
            print("PASSED!!")
        else:
            print("FAILED!")
        print("Expected impact velocity to be %f and it was %f \n" %(V_impact, V))

        break

if ANIMATE:
    input("Press any key to terminate the program")
    if VIDEO:
        video.close()

