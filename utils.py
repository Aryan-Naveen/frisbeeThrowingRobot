import numpy as np
from numpy.linalg import *
import scipy.linalg as LA
import numpy as np
from ikShell import IKshell
from tqdm import tqdm




def readPoses(file_path):
    trajectory = np.genfromtxt(file_path, delimiter=',')
    tt = trajectory[:, 0]
    poses = [T.reshape(4, 4) for T in trajectory[:, 1:]]
    for pose in poses:
    	pose[:3, 3] = 1000*pose[:3, 3]
    return poses, tt


def transformToRobotFrame(poses):
    T_0_robot = np.array([[0,    1,    0,   -694],
                          [0,     0,    1,    655],
                          [1,    0,    0,    448],
                          [0,     0,    0,    1]])

    T_r_camer = np.array([[0,     1,    0,    0],
                          [1,     0,    0,    0],
                          [0,     0,   -1,    0],
                          [0,     0,    0,    1]])
    T = T_0_robot @ T_r_camer @ np.linalg.inv(poses[0])
    poses_robot = [T @ pose for pose in poses]
    return poses_robot


def get_joint_trajectory(poses):
    thinit = np.array([140.17, -25.59, 24.32, 183.23, 222.31, 94.02])
    thguess = np.deg2rad(thinit)
    trajectory_configuration = []
    for i in tqdm(range(len(poses))):
        trajectory_configuration.append(IKshell(poses[i], thguess))
        thguess = trajectory_configuration[-1]


    return trajectory_configuration

def generate_trajectory(file_path):
    poses_c, tt = readPoses(file_path)
    poses_r = transformToRobotFrame(poses_c)
    for pose in poses_r:
        print(pose[:3, 3])
    return get_joint_trajectory(poses_r), tt
