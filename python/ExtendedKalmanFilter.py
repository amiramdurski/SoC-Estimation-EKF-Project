import numpy as np


class EKF(object):
    def __init__(self, F = None, B = None, poly = None, D = None, Q = None, R = None, P = None, x0 = None, r_0 = None):

        if F is None:
            raise ValueError("Set proper system dynamics.")

        self.n    = F.shape[1]

        self.F    = F
        self.B    = 0 if B is None else B
        self.D    = 0 if D is None else D
        self.Q    = np.eye(self.n) if Q is None else Q
        self.R    = np.eye(self.n) if R is None else R
        self.P    = np.eye(self.n) if P is None else P
        self.x    = np.zeros((self.n, 1)) if x0 is None else x0
        self.r_0  = r_0
        self.poly = poly

    def predict(self, u = 0):
        self.x = self.F @ self.x + self.B @ u             # Predict X - next state
        self.P = self.F @ self.P @ self.F.T + self.Q       # Predict P - variance estimation
        return self.x

    def update(self, z, u = None):
        H      = HJacobian_calculation(self.poly, self.x)
        h      = h_calculation(self.poly, self.x, self.r_0, u)

        S      = H @ self.P @ H.T + self.R
        K      = self.P @ H.T @ S.I

        y      = np.subtract(z, h)
        self.x = self.x + K @ y

        I      = np.eye(self.n)
        self.P = np.subtract(I, K @ H) @ self.P


def HJacobian_calculation(poly, x):
    d_poly   = poly.deriv
    d_poly_x = d_poly(x)
    return d_poly_x


def h_calculation(poly, x, r_0, I):
    new_h = poly(x[1]) + x[0] + r_0 * I
    return new_h
