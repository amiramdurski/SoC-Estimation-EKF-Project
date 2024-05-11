import numpy as np
import matplotlib.pyplot as plt
import math as m
from ParseCSV import csv_measurements_to_numpy_array, creating_linear_model
from KalmanFilter import KalmanFilter
from ExtendedKalmanFilter import EKF
import random


def linear_model_soc_estimation(input_path, output_path, r0, q, s, p):

    # Setting the system state variables
    A, B, H, D, delta_V = creating_linear_model(input_path=input_path, output_path=output_path, r0=r0, q=q, s=s, p=p)
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
    with open('C:\python\FinalProject\LinearModel_logfile.txt', 'w') as file:
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
    plt.title('Voltage as function of SoC Prediction - Linear Model')
    plt.xlabel('SOC Prediction')
    plt.ylabel('Voltage Measurement [V]')
    plt.legend()
    plt.show()

def thevenin_model_soc_estimation(r0, r1, c1, time_step, Q_tot):
    # x = [[SoC], [RC voltage]]
    x = np.array([[0.5], [0.0]])

    exp_coeff = m.exp(-time_step / (c1 * r1))

    # state transition model
    F = np.array([[exp_coeff, 0], [0, 1]])

    # control-input model
    B = np.array([[R1 * (1 - exp_coeff)], [time_step / (Q_tot * 3600)]])

    std_dev = 0.015
    # variance from std_dev
    var = std_dev ** 2

    # measurement noise
    R = var

    # state covariance
    P = np.array([[var, 0], [0, var]])

    # process noise covariance matrix
    Q = np.array([[var / 50, 0], [0, var / 50]])

    D    = np.array([[r0]])

    # Define the Polynomial - left is the smallest coefficient
    poly = np.poly1d([4.5056, -4.0621, -13.5688, 24.4140, -14.2391, 3.9905, 3.1400])



def simu_meas_based_model(poly, Q, time_step):
    x = np.linspace(0, 1, 1000)
    y = poly(x)
    plt.figure(figsize=(8, 6))
    plt.plot(x, y)
    plt.title('Polynomial Plot')
    plt.xlabel('x')
    plt.ylabel('P(x)')
    plt.grid(True)
    plt.legend()
    plt.show()

    new_y = np.array([])
    for i in y:
        new_y = np.append(new_y, random.uniform(i*0.995, i*1.005))

    plt.figure(figsize=(8, 6))
    plt.plot(x, new_y)
    plt.title('Polynomial Plot')
    plt.xlabel('x')
    plt.ylabel('P(x)')
    plt.grid(True)
    plt.legend()
    plt.show()

    I = [3.2/1000] * 1000
    I = np.array(I)
    return(I, new_y)


if __name__ == '__main__':
    input_path = 'C:\python\FinalProject\output.csv'  # Path to your input CSV file
    output_path = 'C:\python\FinalProject\gen_output.csv'  # Path to save the modified CSV file
    #linear_model_soc_estimation(input_path=input_path, output_path=output_path, r0=12.5e-3, q=78, s=6, p=4)
    #thevenin_model_soc_estimation(r0=0.062, r1=0.01, c1=3000, time_step=10, Q_tot=3.2)

    poly = np.poly1d([4.5056, -4.0621, -13.5688, 24.4140, -14.2391, 3.9905, 3.1400])
    Q = 3.2*3600
    time_step = 3.6
    simu_meas_based_model(poly=poly, Q=Q, time_step=time_step)
