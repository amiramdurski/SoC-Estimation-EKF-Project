import numpy as np


class KalmanFilter(object):
    def __init__(self, A = None, B = None, H = None, D = None, Q = None, R = None, P = None, x0 = None):

        if A is None or H is None:
            raise ValueError("Set proper system dynamics.")

        self.n = A.shape[1]
        self.m = H.shape[1]

        self.A = A
        self.H = H
        self.B = 0 if B is None else B
        self.D = 0 if D is None else D
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0


    def predict(self, u = 0):
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u)              # Predict X - next state
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q       # Predict P - variance estimation
        return self.x

    def update(self, z, u = None):
        if u is None:
            u  = 0
        y      = z - np.dot(self.H, self.x) - np.dot(self.D, u)
        S      = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K      = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))           # Kalman Gain Calculation
        self.x = self.x + np.dot(K, y)                                   # The Estimation Formula of X - Update X
        I      = np.eye(self.n)                                               # Defining the identity matrix of size n
        self.P = np.dot(I - np.dot(K, self.H), self.P)                   # Update P
        return self.x, self.P

