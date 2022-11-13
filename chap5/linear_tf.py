from model_coef import *

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from control.matlab import *  # MATLAB-like control toolbox functionality


# longitudinal states: u, w, q, theta, h
# inputs: delta_e, delta_t
w, v = LA.eig(A_lon)
#B_lon = B_lon[:,0:0]
print(B_lon.shape) # only elevator input: delta_e
C_lon = np.eye(5)
D_lon = np.zeros((5,2 ))
sys = ss(A_lon, B_lon, C_lon, D_lon)

fig, ax = plt.subplots()
ax.set_xlabel(r'$Re$')
ax.set_ylabel(r'$Im$')
ax.plot(np.real(w), np.imag(w), 'x')
ax.grid()
plt.show()

# Step response for the system
t = np.linspace(0,100,5000)
y, t = impulse(sys, t)
fig, ax = plt.subplots()
color = 'tab:blue'
ax.set_xlabel(r'$t$ [s]')
ax.set_ylabel(r'states')
ax.plot(t, y[:,0,0],label='u')
ax.plot(t, y[:,1,0],label='w')
ax.plot(t, y[:,2,0],label='q')
ax.plot(t, y[:,3,0],label='theta')
#ax.plot(t, y[:,4,0],label='h')
ax.legend()
plt.show()
# Plot t, x for all mode

s = tf('s')
G = (a_phi1)/(s + a_phi2)
