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
    def __init__(self, file_path, vel_control=False):
        self.dt = 1
        self.cTheta = None
        self.k = 0

        self.armCmd = rospy.Publisher('/pos_joint_traj_controller/command', JointTrajectory, queue_size=10, latch=True)
        self.robotCmd = rospy.Publisher('/scaled_pos_joint_traj_controller/command',JointTrajectory,queue_size=10, latch=True)

        if vel_control: self.armCmd = rospy.Publisher('/joint_group_vel_controller/command', Float64MultiArray, queue_size=10, latch=True)
        self.thetas = np.array(u.generate_trajectory(file_path)[:-1]).reshape(42, 6)
        self.n = 42

        rospy.Subscriber('/joint_states', JointState, self.transform_callback)

    def transform_callback(self, msg):
        self.cTheta = np.array(msg.position)
        self.cTheta[0] = np.array(msg.position)[2]
        self.cTheta[2] = np.array(msg.position)[0]

    def compute_target_velocity(self):
        thetadot =  (self.thetas[self.k] - self.cTheta)/self.dt
        self.k += 1
        return thetadot

    def followSpeedTrajectory(self):
        for k in range(self.n):
            thetadot = self.compute_target_velocity()
            velcmd = Float64MultiArray()
            velcmd.data = np.array(thetadot)
            self.armCmd.publish(velcmd)
            rospy.sleep(self.dt)

        velcmd = Float64MultiArray()
        velcmd.data = np.zeros(6, )
        self.armCmd.publish(velcmd)

    def followPositionTrajectory(self):
        for k in range(self.n):
            jointMsg = JointTrajectory()
            jointMsg.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
            jointMsg.points = []
            p = JointTrajectoryPoint()
            p.positions = self.thetas[k]
            print(self.thetas[k])
            p.velocities = [0, 0, 0, 0, 0, 0]
            p.time_from_start.secs = self.dt

            jointMsg.points.append(p)
            self.armCmd.publish(jointMsg)
            self.robotCmd.publish(jointMsg)
            rospy.sleep(self.dt + 1)


if __name__ == '__main__':
    frisbeeThrow = PathFollower('backhand_recording.csv') # mm
    rospy.init_node('frisbee_throw', anonymous=True)
    frisbeeThrow.followPositionTrajectory()
    rospy.spin()
