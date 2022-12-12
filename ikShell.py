import numpy as np
from numpy.linalg import *
import scipy.linalg as LA
import numpy as np


def IKshell(T,theta0):
    # Numerical calculation of UR5e inverse kinematics for end-effector
    # position described by the transformation matrix T, starting from
    # initial guess theta0 for the angles.

    # Make sure theta0 is a column vector
    if theta0.shape == (1,6):
        theta0 = theta0.reshape(6,1)
    elif theta0.shape == (6,1):
        print('Initial guess needs to be a 1D vector')
        return

    # Repeating the arm geometry from the FK lab; all values in mm:
    W2 = 259.6
    W1 = 133.3
    H2 = 99.7
    H1 = 162.5
    L1 = 425
    L2 = 392.2

    # Transformation between the world and body frames when at home position:
    M = np.array(
                [[-1,    0,    0,   L1 + L2],
                 [0,    0,    1,    W1 + W2],
                 [0,    1,    0,    H1 - H2],
                 [0,    0,    0,    1]]
                 )

    # Joint screw axes in the body frame, vector form:
    B1 =  screw_axis(np.array([0, 1, 0]), np.array([L1 + L2, 0, -(W1 + W2)]))
    B2 =  screw_axis(np.array([0, 0, 1]), np.array([L1 + L2, H2, 0]))
    B3 =  screw_axis(np.array([0, 0, 1]), np.array([L2, H2, 0]))
    B4 =  screw_axis(np.array([0, 0, 1]), np.array([0, H2, 0]))
    B5 =  screw_axis(np.array([0, -1, 0]), np.array([0, 0, -W2]))
    B6 =  screw_axis(np.array([0, 0, 1]), np.array([0, 0, 0]))

    B1b =  bracket_screw(B1)
    B2b =  bracket_screw(B2)
    B3b =  bracket_screw(B3)
    B4b =  bracket_screw(B4)
    B5b =  bracket_screw(B5)
    B6b =  bracket_screw(B6)



    # Here begins the iterative algorithm described in the textbook,
    # in the form described starting "To modify this algorithm to work
    # with a desired end-effector configuration represented as T_sd...":

    thguess = theta0  # initialize the current guess to the user-supplied value
    lastguess = thguess * 10 + 50  # arbitrary value far away from the initial guess, to ensure the while loop is entered
    # Termination condition, indicating the algorithm has converged:
    i = 0
    while np.linalg.norm(thguess-lastguess,2) > 1e-3:
        lastguess = thguess

        # Step (b) of the iterative algorithm is:
        # "Set [V_b] = log(T^{-1}_{ab}(theta^i)T_{ad})"
        # We can go about that as follows:

        # From above, we have the joint screw axes in the body frame; now for
        # each joint, convert from the exponential coordinate representation
        # for the rotation around the screw axis ((B,theta) form) to the 4x4
        # matrix representation of the transformation ((R,d) form):

        eB1 =  screw_exp(B1, thguess[0])
        eB2 =  screw_exp(B2, thguess[1])
        eB3 =  screw_exp(B3, thguess[2])
        eB4 =  screw_exp(B4, thguess[3])
        eB5 =  screw_exp(B5, thguess[4])
        eB6 =  screw_exp(B6, thguess[5])

        # Each of the columns of the Jacobian is the vector form of the above:
        T_B1 = inv(eB6) @ inv(eB5) @ inv(eB4) @ inv(eB3) @ inv(eB2)
        AD_B1 =  adjoint_matrix(T_B1);

        T_B2 = inv(eB6) @ inv(eB5) @ inv(eB4) @ inv(eB3)
        AD_B2 =  adjoint_matrix(T_B2);

        T_B3 = inv(eB6) @ inv(eB5) @ inv(eB4)
        AD_B3 =  adjoint_matrix(T_B3);

        T_B4 = inv(eB6) @ inv(eB5)
        AD_B4 =  adjoint_matrix(T_B4);

        T_B5 = inv(eB6)
        AD_B5 =  adjoint_matrix(T_B5);

        J1 = AD_B1 @ B1
        J2 = AD_B2 @ B2
        J3 = AD_B3 @ B3
        J4 = AD_B4 @ B4
        J5 = AD_B5 @ B5

        J6 = B6


        # Finally we can assemble the complete Jacobian:
        J = np.bmat([J1.reshape(6, 1), J2.reshape(6, 1), J3.reshape(6, 1), J4.reshape(6, 1), J5.reshape(6, 1), J6.reshape(6, 1)])

        # Forward kinematics for the robot's pose as a function of theta:
        Tab = M @ eB1 @ eB2 @ eB3 @ eB4 @ eB5 @ eB6

        # Convert the transformation matrix we want (the input T) into the body frame:
        Tbd = np.linalg.inv(Tab) @ T


        # Calculate the matrix logarithm of T_bd:
        # Finally we can set [Vb] equal to that log, which is what we set out to do:
        # if thguess[0] == 0:
        #     print(mr.TransToRp(Tbd))
        Vbb, theta,W =  logm(Tbd)


        # Convert the matrix form of the body twist to the vector form, [Vb] -> Vb:
        Vb =  skew_to_vector(Vbb)
        # The last step is to update the current guess for the angles, using
        # "Set theta^{i+1} = theta^{i} + pinv(Jb(theta^{i}))*Vb":
        thguess = (lastguess + np.linalg.pinv(J) @ Vb).tolist()[0]
        for i in range(6):
            if isinstance(thguess[i], complex):
                thguess[i] = thguess[i].real
        thguess = np.array(thguess)


        i = i +  1
    # Once the algorithm has converged, return the final angle found:
    theta = thguess

    # return theta
#    print(theta)
#    print(theta[1] % 2*np.pi)
    theta = [th % (2*np.pi) for th in theta]
    if theta[1] > np.pi/4:
        theta[1] = theta[1] - 2*np.pi
    return np.array(theta)


def DH(th):
    # Uses the Denavit-Hartenberg method to determine the final position of
    # the end effector using the joint positions of the robot.
    # th is a 6x1 matrix describing the robot's joint positions.

    # The given dimensions of the robot:
    W2 = 259.6
    W1 = 133.3
    H2 = 99.7
    H1 = 162.5
    L1 = 425
    L2 = 392.2

    # When you work through the frame assignment procedure discussed in class,
    # you'll come up with 4 parameters for each of the 6 joints. It can be
    # convenient to put these parameter values into a single table, where
    # each row gives the four parameters for one link. (Both the Lynch/Park
    # and Spong textbooks give examples of such tables for various robots.)
    # Create a 6x4 matrix of those values here:

    DHmat = np.empty([6, 4])
    # theta d a alpha
    DHmat[0] = [th[0], H1, 0, -np.pi/2]
    DHmat[1] = [th[1], 0, L1, 0]
    DHmat[2] = [th[2], 0, L2, 0]
    DHmat[3] = [th[3], W1, 0, -np.pi/2]
    DHmat[4] = [th[4], H2, 0, np.pi/2]
    DHmat[5] = [th[5] - np.pi/2, W2, 0, 0]

    # Since in the D-H formulation, the transformation that occurs at each
    # joint is a product of four simpler transformations (each a pure
    # translation or rotation, with respect to a single coordinate axis),
    # create helper functions for performing those simpler transformations:
    def RX(thet):
        Tret = np.array([
            [1,     0,              0,              0],
            [0,     np.cos(thet),   -np.sin(thet),  0],
            [0,     np.sin(thet),   np.cos(thet),   0],
            [0,     0,              0,              1]
        ])

        return Tret

    def RZ(thet):
        Tret = np.array([
            [np.cos(thet),  -np.sin(thet),  0,   0],
            [np.sin(thet),  np.cos(thet),   0,   0],
            [0,             0,              1,   0],
            [0,             0,              0,   1]
        ])
        return Tret

    def TX(dist):
        Tret = np.array([
            [1, 0, 0, dist],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        return Tret

    def TZ(dist):
        Tret = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dist],
            [0, 0, 0, 1]
        ])
        return Tret

    # Now you can use those functions and the D-H table to calculate the
    # transformation matrix for each successive joint relative to the last:
    T1 = RZ(DHmat[0, 0]) @ TZ(DHmat[0, 1]) @ TX(DHmat[0, 2]) @ RX(DHmat[0, 3])
    T2 = RZ(DHmat[1, 0]) @ TZ(DHmat[1, 1]) @ TX(DHmat[1, 2]) @ RX(DHmat[1, 3])
    T3 = RZ(DHmat[2, 0]) @ TZ(DHmat[2, 1]) @ TX(DHmat[2, 2]) @ RX(DHmat[2, 3])
    T4 = RZ(DHmat[3, 0]) @ TZ(DHmat[3, 1]) @ TX(DHmat[3, 2]) @ RX(DHmat[3, 3])
    T5 = RZ(DHmat[4, 0]) @ TZ(DHmat[4, 1]) @ TX(DHmat[4, 2]) @ RX(DHmat[4, 3])
    T6 = RZ(DHmat[5, 0]) @ TZ(DHmat[5, 1]) @ TX(DHmat[5, 2]) @ RX(DHmat[5, 3])
    # Next calculate the final transformation which is the result of
    # performing the six separate transformations in succession:
    Tfinal = T1 @ T2 @ T3 @ T4 @ T5 @ T6

    return Tfinal


def screw_axis(w, q):
    v = np.cross(-w, q)
    return np.bmat([w, v]).T

def bracket_screw(S):
    W = skew(S[:3].reshape(3, ))
    c = S[3: ].reshape(3, 1)
    return np.bmat([[W, c], [np.zeros((1, 4))]])

def skew(x):
    x_f = np.array(x).reshape(3, )
    return np.array([[0, -x_f[2], x_f[1]],
                     [x_f[2], 0, -x_f[0]],
                     [-x_f[1], x_f[0], 0]])


def screw_exp(B, theta):
    W = skew(B[:3])
    v = np.array(B[3:]).reshape(3, 1)
    # print(np.eye(3) + np.sin(-theta)*W + (1- np.cos(-theta))* W @ W)
    e_minus_W_t = np.eye(3) + np.sin(theta)*W + (1- np.cos(theta))* W @ W
    G_minus_t = theta*np.eye(3) + (1 - np.cos(theta))*W + (theta - np.sin(theta))*W@W
    G_t_v = G_minus_t @ v
    A = np.bmat([[e_minus_W_t, G_t_v], [np.zeros((1, 3)), np.ones((1, 1))]])
    return A

def adjoint_matrix(T):
    R = T[:3, :3]
    pos_R = skew(T[:3, 3].ravel()) @  T[:3, :3]
    A = np.bmat([[R, np.zeros((3, 3))], [pos_R, R]])
    return A

def logm(A):
    R = A[:3, :3]
    p = A[:3, 3]
    if np.linalg.norm(R - np.eye(3)) < 1e-8:
        w = np.zeros(3, )
        v = p/np.linalg.norm(p)
        theta = np.linalg.norm(p)
        W = skew(w)
    else:
        if np.abs(np.trace(R) + 1) < 1e-8:
            theta = np.pi
            if np.abs(R[0, 0] + 1) > 1e-8:
                w = 1/np.sqrt(2*(1 + R[0, 0])) * np.array([1 + R[0, 0], R[1, 0], R[2, 0]])
            elif np.abs(R[1, 1] + 1) > 1e-8:
                w = 1/np.sqrt(2*(1 + R[1, 1])) * np.array([R[0, 1], 1 + R[1, 1], R[2, 1]])
            else:
                w = 1/np.sqrt(2*(1 + R[2, 2])) * np.array([R[0, 2], R[1, 2], 1 + R[2, 2]])
            W = skew(w)
        else:
            val = 0.5*(np.trace(R) - 1)
            theta = np.emath.arccos(val)
            W = (1/2/np.sin(theta))*(R - R.T)

        G_inv_theta = (1/theta)*np.eye(3) - 0.5*W + (1/theta - 0.5*(1/np.tan(theta/2)))*W @ W
        v = G_inv_theta @ p

    return np.bmat([[theta*W, (v*theta).reshape(3, 1)], [np.zeros((1, 4))]]), theta, W

def skew_to_vector(X):
    return [X[2,1], X[0,2], X[1,0], X[0,3], X[1,3], X[2,3]]
