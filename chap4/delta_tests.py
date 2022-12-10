"""
mavsimPy
    - Chapter 4 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        12/27/2018 - RWB
        1/17/2019 - RWB
"""
import sys
import os

sys.path.append(os.path.split(sys.path[0])[0])
# from tools.rotations import Euler2Quaternion, Euler2Rotation, Quaternion2Rotation

import numpy as np
import parameters.simulation_parameters as SIM

from chap2.mav_viewer import MavViewer
from chap3.data_viewer import DataViewer
from chap4.mav_dynamics import MavDynamics
from chap4.wind_simulation import WindSimulation
from message_types.msg_delta import MsgDelta

# initialize the visualization
VIDEO = False  # True==write video, False==don't write video
mav_view = MavViewer()  # initialize the mav viewer
data_view = DataViewer()  # initialize view of data plots
if VIDEO is True:
    from chap2.video_writer import VideoWriter
    video = VideoWriter(video_name="chap4_video.mp4",
                        bounding_box=(0, 0, 1000, 1000),
                        output_rate=SIM.ts_video)

# initialize elements of the architecture
wind = WindSimulation(SIM.ts_simulation)
delta = MsgDelta()

delta_matrix = np.array([
    [0, 0, 0, 0],
    [0.25, 0, 0, 0],
    [-0.25, 0, 0, 0],
    [0, 0.25, 0, 0],
    [0, -0.25, 0, 0],
    [0, 0, 0.25, 0],
    [0, 0, -0.25, 0],
    ])

for thing in delta_matrix:
    delta.from_array(thing)

    mav = MavDynamics(SIM.ts_simulation)
    # initialize the simulation time
    sim_time = SIM.start_time
    plot_time = sim_time

    # main simulation loop
    while sim_time < SIM.end_time:
        # -------physical system-------------
        current_wind = wind.update()  # get the new wind vector
        mav.update(delta, current_wind)  # propagate the MAV dynamics

        # -------update viewer-------------
        if sim_time-plot_time > SIM.ts_plotting:
            mav_view.update(mav.true_state)  # plot body of MAV
            plot_time = sim_time
        data_view.update(mav.true_state,  # true states
                        mav.true_state,  # estimated states
                        mav.true_state,  # commanded states
                        delta,  # inputs to aircraft
                        SIM.ts_simulation)

        # -------increment time-------------
        sim_time += SIM.ts_simulation

input("Press any key to terminate the program")


# from PyQt5 import QtGui  # (the example applies equally well to PySide)
# from PyQt5 import QtWidgets  # (the example applies equally well to PySide)
# import pyqtgraph as pg
# import pyqtgraph.opengl as gl

# ## Always start by initializing Qt (only once per application)
# app = QtWidgets.QApplication([])

# ## Define a top-level widget to hold everything
# w = QtWidgets.QWidget()
# w.resize(1000,600)


# ## Create some widgets to be placed inside
# btn = QtWidgets.QPushButton('press me')
# text = QtWidgets.QLineEdit('enter text')
# listw = QtWidgets.QListWidget()
# # plot = gl.GLViewWidget()
# # plot.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
# # g = gl.GLGridItem()
# # plot.addItem(g)


# ## Create a grid layout to manage the widgets size and position
# layout = QtWidgets.QGridLayout()
# w.setLayout(layout)

# ## Add widgets to the layout in their proper positions
# layout.addWidget(btn, 0, 0)   # button goes in upper-left
# layout.addWidget(text, 1, 0)   # text edit goes in middle-left
# layout.addWidget(listw, 2, 0)  # list widget goes in bottom-left
# # layout.addWidget(plot, 0, 1, 3, 1)  # plot goes on right side, spanning 3 rows

# ## Display the widget as a new window
# w.show()

# ## Start the Qt event loop
# app.exec_()

# from PyQt5 import QtCore, QtGui, QtWidgets
# import sys
  
# class Ui_MainWindow(object):
  
#     def setupUi(self, MainWindow):
  
#         MainWindow.resize(550, 393)
#         self.centralwidget = QtWidgets.QWidget(MainWindow)
#         self.centralwidget.setObjectName("centralwidget")
  
#         self.slider = QtWidgets.QSlider(self.centralwidget)
#         self.slider.setGeometry(QtCore.QRect(190, 100, 160, 16))
#         self.slider.setOrientation(QtCore.Qt.Horizontal)
  
#         # After each value change, slot "scaletext" will get invoked.
#         self.slider.valueChanged.connect(self.scaletext)
  
#         self.label = QtWidgets.QLabel(self.centralwidget)
#         self.label.setGeometry(QtCore.QRect(230, 150, 301, 161))
          
#         # set initial font size of label.
#         self.font = QtGui.QFont()
#         self.font.setPointSize(7)
#         self.label.setFont(self.font)
#         MainWindow.setCentralWidget(self.centralwidget)
  
#         self.retranslateUi(MainWindow)
#         QtCore.QMetaObject.connectSlotsByName(MainWindow)
  
#     def retranslateUi(self, MainWindow):
#         _translate = QtCore.QCoreApplication.translate
#         MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
#         self.label.setText(_translate("MainWindow", "QSlider"))
          
#     def scaletext(self, value):
#         # Change font size of label. Size value could 
#         # be anything consistent with the dimension of label.
#         # self.font.setPointSize(7 + value//2)
#         # self.label.setFont(self.font)
#         print(value)
      
      
# if __name__ == "__main__": 
#     app = QtWidgets.QApplication(sys.argv) 
    
#     MainWindow = QtWidgets.QMainWindow() 
#     ui = Ui_MainWindow() 
#     ui.setupUi(MainWindow) 
#     MainWindow.show() 
#     sys.exit(app.exec_()) 