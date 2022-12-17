import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt

import parameters.simulation_parameters as SIM

from chap2.mav_viewer import MavViewer
from chap3.data_viewer import DataViewer
from chap4.mav_dynamics import MavDynamics
from chap4.wind_simulation import WindSimulation
from chap5.trim import compute_trim
from chap5.compute_models import compute_model
from tools.signals import Signals
from message_types.msg_delta import MsgDelta

delta = MsgDelta(throttle=0)
mav = MavDynamics(SIM.ts_simulation)
# use compute_trim function to compute trim state and trim input

throttles = np.arange(0, 1.0, 0.05)
forces_x = np.zeros(throttles.shape)

for (i, throttle) in enumerate(throttles):
    delta.throttle=throttle
    forces = mav._forces_moments(delta)
    forces_x[i] = forces.item(0)
    # print(forces)
    # print('Throttle\t', throttle, '\t Fx \t', forces.item(0))

plt.figure()
plt.plot(throttles, forces_x)
plt.title('Fx v Throttle')
plt.xlabel('Delta T')
plt.ylabel('Net Fx')
plt.show()