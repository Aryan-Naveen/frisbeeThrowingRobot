import numpy as np
import scipy.linalg as sc
import utils as u


import rosnode
import rospy

from std_msgs.msg import String, Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from gazebo_msgs.msg import LinkStates
from sensor_msgs.msg import JointState

from tabulate import tabulate
from tqdm import tqdm
import tf




class PathFollower():
    def __init__(self, file_path):
        self.dt = 0.033
        self.cTheta = None
        self.k = 0
        self.armCmd = rospy.Publisher('/joint_group_vel_controller/command', Float64MultiArray, queue_size=10, latch=True)
        self.thetas = u.generate_trajectory(file_path)

        rospy.Subscriber('/joint_states', JointState, self.transform_callback)

    def transform_callback(self, msg):
        self.cTheta = np.array(msg.position)
        self.cTheta[0] = np.array(msg.position)[2]
        self.cTheta[2] = np.array(msg.position)[0]

    def compute_target_velocity(self):
        thetadot =  (self.thetas[self.k] - self.cTheta)/self.dt
        self.k += 1
        return thetadot

    def followTrajectory(self):
        for k in range(self.n):
            thetadot = self.compute_target_velocity()
            velcmd = Float64MultiArray()
            velcmd.data = np.array(thetadot)
            self.armCmd.publish(velcmd)
            rospy.sleep(self.dt)

        velcmd = Float64MultiArray()
        velcmd.data = np.zeros(6, )
        self.armCmd.publish(velcmd)



if __name__ == '__main__':
    frisbeeThrow = PathFollower('backhand_recording.csv') # mm
    rospy.init_node('frisbee_throw', anonymous=True)
    frisbeeThrow.followTrajectory()
    rospy.spin()
