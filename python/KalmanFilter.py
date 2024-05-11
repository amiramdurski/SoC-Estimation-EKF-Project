import numpy as np
from ParseCSV import csv_measurements_to_numpy_array, creating_physical_model
import matplotlib.pyplot as plt


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
            u = 0
        y = z - np.dot(self.H, self.x) - np.dot(self.D, u)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))           # Kalman Gain Calculation
        self.x = self.x + np.dot(K, y)                                   # The Estimation Formula of X - Update X
        I = np.eye(self.n)                                               # Defining the identity matrix of size n
        self.P = np.dot(I - np.dot(K, self.H), self.P)                   # Update P
        return (self.x, self.P)


def linear_model_soc_estimation(input_path, output_path, r_0, Q, s, p):

    # Setting the system state variables
    A, B, H, D, delta_V = creating_physical_model(input_path=input_path, output_path=output_path, r_0=r_0, Q=Q, s=s, p=p)
    A = np.array([[A]])
    B = np.array([[-B]])
    H = np.array([[H]])
    D = np.array([[D]])
    Q = np.array([[10**-5]])
    R = np.array([[0.1]])
    soc0 = np.array([[0.7]])

    measurements = csv_measurements_to_numpy_array(input_path)  # Define vectors of [i_n, v_n]
    kf = KalmanFilter(A = A, B = B, D = D, H = H, Q = Q, R = R, x0 = soc0)
    predictions = []
    soc_prediction = np.zeros(0)

    count = 0
    with open('C:\python\FinalProject\logfile.txt', 'w') as file:
        file.write(f"The values of the model: \n A = {float(A)}\n B = {float(B)}\n H = {float(H)}\n D = {float(D)}\n Q = {float(Q)}\n R = {float(R)}\n soc0 = {float(soc0)}\n delta_V = {float(delta_V)}\n\n")
        for meas in measurements:
            meas[1] -= delta_V
            count += 1
            I = meas[0]
            V = meas[1]
            predictions.append(np.dot(H,  kf.predict(u = I)))
            x_n, p_n = kf.update(z = V, u = I)
            if np.isnan(x_n):
                soc_prediction = np.append(soc_prediction, 0)
                print("NaN value")
            else:
                soc_prediction = np.append(soc_prediction, x_n)
            line = ("Iteration " + str(float(count)) + " x = " + str(float(x_n)) + ", p = " + str(float(p_n)) + "\n")
            file.write(line)


    # Calculate linear regression (ordinary least squares)
    X = np.vstack([soc_prediction, np.ones(len(soc_prediction))]).T
    m, c = np.linalg.lstsq(X, measurements[:, 1], rcond=None)[0]

    # Plot regular and linear trendline
    plt.plot(soc_prediction, measurements[:, 1], label='Regular Trendline', alpha=0.5)
    plt.plot(soc_prediction, m * soc_prediction + c, 'r-', label='Linear Trendline')

    # Add equation of the linear trendline
    equation_text = f'y = {m:.2f}x + {c:.2f}'
    plt.text(0.5, 0.9, equation_text, fontsize=12, transform=plt.gca().transAxes)

    # Add title, labels, and legend
    plt.title('Voltage as function of SoC Prediction')
    plt.xlabel('SOC Prediction')
    plt.ylabel('Voltage Measurment [V]')
    plt.legend()
    plt.show()

def ex6_q5_example():

    x0 = np.array([[20]])
    P  = np.array([[4]])
    Q  = np.array([[0.01]])
    H  = np.array([[1]])
    A  = np.array([[1]])
    R  = np.array([[2.25]])
    #measurements = np.array([[24], [26]])
    measurements = np.array([[23.58], [23.12], [24.31], [27.17], [24.94], [24.23], [26.04], [24.47], [24.65], [24.55]])
    kf = KalmanFilter(A = A, H = H, Q = Q, R = R, x0 = x0, P = P)
    predictions = []

    count = 0
    for meas in measurements:
        count += 1
        predictions.append(np.dot(H,  kf.predict()))
        x_n, p_n = kf.update(z = meas)
        print("x for", count,"iteration: ", x_n)
        print("p for", count,"iteration: ", p_n)


if __name__ == '__main__':
    #ex6_q5_example()
    input_path = 'C:\python\FinalProject\output.csv'  # Path to your input CSV file
    output_path = 'C:\python\FinalProject\gen_output.csv'  # Path to save the modified CSV file
    linear_model_soc_estimation(input_path=input_path, output_path=output_path, r_0=12.5e-3, Q=78, s=6, p=4)

"""
    A = np.array([[1]])
    B = np.array([[-0.03561e-6]])
    H = np.array([[5.3869]])
    D = np.array([[-0.01]])
    delta_V = 19.29
"""
