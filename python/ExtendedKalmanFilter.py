import numpy as np
import inspect
from itertools import product
from ParseCSV import ModelType

def custom_print(*args, **kwargs):
    frame = inspect.currentframe().f_back
    filename = frame.f_code.co_filename
    lineno = frame.f_lineno
    print(f"{filename}:{lineno}:", *args, **kwargs)

def get_caller_info():
    frame = inspect.currentframe().f_back
    filename = frame.f_code.co_filename
    lineno = frame.f_lineno
    return filename, lineno
def CheckVariableDimension(var, var_name, dimension_rint, dimension_thevenin, model_type, filename=None, lineno=None):
    frame = inspect.currentframe().f_back
    filename = frame.f_code.co_filename
    lineno = frame.f_lineno
    if model_type == 'RintModel' and var.shape != dimension_rint:
        print(f"WRONG DIMENSION:{var_name} dimension is {var.shape} instead of {dimension_rint}, var_value = {var}, ModelType = {model_type}")
        if filename != None and lineno != None:
            print(f"Called from file: {filename}, line: {lineno}")
        exit(1)
    elif model_type == 'TheveninModel' and var.shape != dimension_thevenin:
        print(f"WRONG DIMENSION: {var_name} dimension is {var.shape} instead of {dimension_thevenin}, var_value = {var}, ModelType = {model_type}")
        if filename != None and lineno != None:
            print(f"Called from file: {filename}, line: {lineno}")
        exit(1)

class EKF(object):
    def __init__(self, F=None, B = None, D = None, Q = None, R = None, P = None, x0=None, poly=None, model_type=None, coefficients=None, soc_degree=None, temperature_degree=None):

        if F is None:
            raise ValueError("Set proper system dynamics.")

        self.n                  = F.shape[1]
        self.F                  = F
        self.B                  = 0 if B is None else B
        self.D                  = 0 if D is None else D
        self.Q                  = np.eye(self.n) if Q is None else Q
        self.R                  = np.eye(self.n) if R is None else R
        self.P                  = np.eye(self.n) if P is None else P
        self.x                  = np.zeros((self.n, 1)) if x0 is None else x0
        self.poly               = poly
        self.model_type         = model_type
        self.coefficients       = coefficients
        self.soc_degree         = soc_degree
        self.temperature_degree = temperature_degree

    def predict(self, u=0):
        #u = u.reshape(self.B.shape)
        filename, lineno = get_caller_info()
        CheckVariableDimension(var=self.x, var_name="x", dimension_rint=(1, 1), dimension_thevenin=(2, 1), model_type=self.model_type, filename=filename, lineno=lineno)
        filename, lineno = get_caller_info()
        CheckVariableDimension(var=self.F, var_name="F", dimension_rint=(1, 1), dimension_thevenin=(2, 2), model_type=self.model_type, filename=filename, lineno=lineno)
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)              # Predict X - next state
        #self.x = self.F @ self.x + self.B @ u             # Predict X - next state
        #self.P = self.F @ self.P @ self.F.T + self.Q      # Predict P - variance estimation
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q      # Predict P - variance estimation

        return self.x

    def update(self, z, u=None, Temperature=None):
        H      = self.HJacobian_calculation(Temperature)
        h      = self.h_calculation(u, Temperature)

        #S      = H @ self.P @ H.T + self.R
        S      = np.dot(H, np.dot(self.P, H.T)) + self.R

        filename, lineno = get_caller_info()
        CheckVariableDimension(var=self.P, var_name="P", dimension_rint=(1, 1), dimension_thevenin=(2, 2), model_type=self.model_type, filename=filename, lineno=lineno)
        CheckVariableDimension(var=H, var_name="H", dimension_rint=(1, 1), dimension_thevenin=(1, 2), model_type=self.model_type, filename=filename, lineno=lineno)
        CheckVariableDimension(var=self.R, var_name="R", dimension_rint=(1, 1), dimension_thevenin=(1, 1), model_type=self.model_type, filename=filename, lineno=lineno)
        CheckVariableDimension(var=S, var_name="S", dimension_rint=(1, 1), dimension_thevenin=(1, 1), model_type=self.model_type, filename=filename, lineno=lineno)

        K      = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))
        #K      = self.P @ H.T @ np.linalg.inv(S)
        y      = np.subtract(z, h[0])
        filename, lineno = get_caller_info()
        CheckVariableDimension(var=K, var_name="K", dimension_rint=(1, 1), dimension_thevenin=(2, 1), model_type=self.model_type, filename=filename, lineno=lineno)
        CheckVariableDimension(var=self.x, var_name="x", dimension_rint=(1, 1), dimension_thevenin=(2, 1), model_type=self.model_type, filename=filename, lineno=lineno)
        self.x = self.x + np.dot(K, y)
        filename, lineno = get_caller_info()
        CheckVariableDimension(var=self.x, var_name="x", dimension_rint=(1, 1), dimension_thevenin=(2, 1), model_type=self.model_type, filename=filename, lineno=lineno)


        I      = np.eye(self.n)
        #self.P = np.subtract(I, K @ H) @ self.P
        self.P = np.dot(np.subtract(I, np.dot(K, H)), self.P)

        #print(f"self.x = {self.x}, type(self.x) = {type(self.x)} ,self.x.shape = {self.x.shape}")
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

    def HJacobian_calculation(self, Temperature):
        if (self.temperature_degree != 0):
            d_poly_x_t = self.evaluate_derivative_soc(self.x[1], Temperature)
            return np.array([[-1, d_poly_x_t[0]]])
        else:
            d_poly   = self.poly.deriv()
            if self.model_type == "RintModel":
                d_poly_x = d_poly(self.x)
                return d_poly_x
            elif self.model_type == "TheveninModel":
                d_poly_x = d_poly(self.x[1])
                d_poly_x = float(d_poly_x[0])
                return np.array([[-1, d_poly_x]])

    def h_calculation(self, I, Temperature):
        if (self.temperature_degree != 0):
            new_h = self.evaluate_polynomial(self.x[1], Temperature) + self.x[0] + self.D * I
            return new_h
        else:
            if self.model_type == "RintModel":
                new_h = self.poly(self.x) + self.D * I
            elif self.model_type == "TheveninModel":
                new_h = self.poly(self.x[1]) + self.x[0] + self.D * I
            return new_h

    def evaluate_polynomial(self, soc, temp):
        result = 0
        index = 0
        #max_degree = max(self.soc_degree, self.temperature_degree) + 1
        #comb = list(combinations_with_replacement(range(max_degree), 2))
        comb = list(product(range(self.soc_degree + 1), range(self.temperature_degree + 1)))
        for i_soc, i_temp in comb:
            if index >= len(self.coefficients):
                break
            result += self.coefficients[index] * (soc ** i_soc) * (temp ** i_temp)
            index += 1
        return result

    def evaluate_derivative_soc(self, soc, temp):
        #print(f"IN EKF: Soc= {soc}, temp={temp}")
        result = 0
        index = 0
        #max_degree = max(self.soc_degree, self.temperature_degree) + 1
        #comb = list(combinations_with_replacement(range(max_degree), 2))
        comb = list(product(range(self.soc_degree + 1), range(self.temperature_degree + 1)))
        for i_soc, i_temp in comb:
            if index >= len(self.coefficients):
                break
            if i_soc > 0:
                result += self.coefficients[index] * i_soc * (soc ** (i_soc - 1)) * (temp ** i_temp)
            index += 1
        return result
