import numpy as np
from numpy.linalg import *
import scipy.linalg as LA
import numpy as np
from ikShell import IKshell


T_0_robot = [[0, -1, 0, -215], [0, 0, 1, 850], [-1, 0, 0, 600], [0, 0, 0, 1]]

def readPoses(file_path):
    trajectory = np.genfromtxt(file_path, delimiter=',')
    tt = trajectory[:, 0]
    poses = [T.reshape(4, 4) for T in trajectory[:, 1:]]
    return poses, tt


def transformToRobotFrame(poses):
    T_0_robot = np.array([[0,    -1,    0,   -215],
                          [0,     0,    1,    850],
                          [-1,    0,    0,    600],
                          [0,     0,    0,    1]])
    T = T_0_robot @ np.linalg.inv(poses[0])
    poses_robot = [T @ pose for pose in poses]
    return poses_robot

def get_joint_trajectory(poses):
    thguess = np.array([0, 0, 0, 0, 0, 0])
    trajectory_configuration = []
    for i in tqdm(range(len(poses))):
        trajectory_configuration.append(IKshell(poses[i], thguess))
        thguess = trajectory_configuration[-1]

    return trajectory_configuration

def generate_trajectory(file_path):
    poses_c, tt = readPoses(file_path)
    poses_r = transformToRobotFrame(poses_c)
    return get_joint_trajectory(poses_r), tt
