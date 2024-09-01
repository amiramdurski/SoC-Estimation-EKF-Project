import numpy as np

def CheckVariableDimension(var, var_name, dimension_rint, dimension_thevenin, model_type):
    if model_type == 'RintModel' and var.shape != dimension_rint:
        print(f"WRONG DIMENSION: {var_name} dimension is {var.shape} instead of {dimension_rint}, var_value = {var}, ModelType = {model_type}")
        exit(1)
    elif model_type == 'TheveninModel' and var.shape != dimension_thevenin:
        print(f"WRONG DIMENSION: {var_name} dimension is {var.shape} instead of {dimension_thevenin}, var_value = {var}, ModelType = {model_type}")
        exit(1)


class KalmanFilter(object):
    def __init__(self, A = None, B = None, H = None, D = None, Q = None, R = None, P = None, x0 = None, model_type = None):

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
        self.model_type = model_type


    def predict(self, u = 0):
        CheckVariableDimension(var=self.x, var_name="x", dimension_rint=(1, 1), dimension_thevenin=(2, 1), model_type=self.model_type)
        CheckVariableDimension(var=self.A, var_name="A", dimension_rint=(1, 1), dimension_thevenin=(2, 2), model_type=self.model_type)
        CheckVariableDimension(var=self.B, var_name="B", dimension_rint=(1, 1), dimension_thevenin=(2, 1), model_type=self.model_type)
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u)              # Predict X - next state
        CheckVariableDimension(var=self.x, var_name="x", dimension_rint=(1, 1), dimension_thevenin=(2, 1), model_type=self.model_type)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q       # Predict P - variance estimation
        CheckVariableDimension(var=self.P, var_name="P", dimension_rint=(1, 1), dimension_thevenin=(2, 2), model_type=self.model_type)
        #return self.x

    def update(self, z, u=None, Temperature=None):
        if u is None:
            u  = 0
        y      = z - np.dot(self.H, self.x) - np.dot(self.D, u)
        S      = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K      = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))      # Kalman Gain Calculation
        self.x = self.x + np.dot(K, y)                                   # The Estimation Formula of X - Update X
        I      = np.eye(self.n)                                          # Defining the identity matrix of size n
        self.P = np.dot(I - np.dot(K, self.H), self.P)                   # Update P

        #print(f"self.x = {self.x}, type(self.x) = {type(self.x)} ,self.x.shape = {self.x.shape} , model_type = {self.model_type}")
        if self.model_type == "RintModel":
            if self.x > 1:
                #print("ValueError: SOC is bigger than 100%")
                self.x = np.array(0.9999).reshape(1, 1)
            elif self.x < 0:
                #print("ValueError: SOC is smaller than 0%")
                self.x = np.array(0.0001).reshape(1, 1)
        else:
            if self.x[1] > 1:
                #print("ValueError: SOC is bigger than 100%")
                self.x[1] = np.array(0.9999).reshape(1, 1)
            elif self.x[1] < 0:
                #print("ValueError: SOC is smaller than 0%")
                self.x[1] = np.array(0.0001).reshape(1, 1)
        return self.x, self.P



