import numpy as np
from scipy.spatial.transform import Rotation as R
import scipy.signal as ss
import matplotlib.pyplot as plt
from scipy import integrate

def deviceFrameToFixedFrame(quat, accel):
    R_ = R.from_quat(quat).as_matrix()
    return R_ @ accel.T



data_dir = 'data/Nov29_linemotion/'

if __name__ == '__main__':
    orientation_signal = np.genfromtxt(data_dir + 'Orientation.csv', delimiter=',')
    acceleration_signal = np.genfromtxt(data_dir + 'Gyroscope.csv', delimiter=',')
    tt = acceleration_signal[1:, 1].T

    qq = np.apply_along_axis(ss.medfilt, 0, orientation_signal[1:, [3, 7, 4, 6]])
    aa = np.apply_along_axis(ss.medfilt, 0, acceleration_signal[1:, [4, 3, 2]])
    aa = aa - np.array([ 0.0241167,  -0.01986794,  0.02620967])

    print(aa[:, 2])
    transformToFixedFrame = np.vectorize(deviceFrameToFixedFrame)

    def transformToFixedFrame(quaternions, acceleration):
        for ind_ in range(quaternions.shape[0]):
            acceleration[ind_] = deviceFrameToFixedFrame(quaternions[0], acceleration[ind_])
        return acceleration


    # aa = transformToFixedFrame(qq, aa)

    vv = integrate.cumtrapz(aa, x=tt, axis=0, initial = 0)
    xx = integrate.cumtrapz(vv, x=tt, axis=0, initial = 0)

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(xx[:, 0], xx[:, 1], xx[:, 2]);
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    fig, ax = plt.subplots(1, 1, figsize=(18, 18))
    ax.plot(tt, aa[:, 0], label="ax")
    ax.plot(tt, aa[:, 1], label="ay")
    ax.plot(tt, aa[:, 2], label="az")
    ax.legend()
    plt.show()
