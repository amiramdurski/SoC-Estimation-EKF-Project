import numpy as np
import matplotlib.pyplot as plt
from ParseCSV import CsvMeasToNumpyArray, CreateBatteryModel, SliceFileName
from KalmanFilter import KalmanFilter
from ExtendedKalmanFilter import EKF
import os
import pandas as pd
from ParseCSV import ModelType
from itertools import product
from scipy.signal import savgol_filter
import json
import csv
import math as m
import random



def CheckVariableDimension(var, var_name, dimension_rint, dimension_thevenin, model_type):
    if model_type == 'RintModel' and var.shape != dimension_rint:
        print(f"WRONG DIMENSION: {var_name} dimension is {var.shape} instead of {dimension_rint}, var_name = {var}, ModelType = {model_type}")
    elif model_type == 'TheveninModel' and var.shape != dimension_thevenin:
        print(f"WRONG DIMENSION: {var_name} dimension is {var.shape} instead of {dimension_thevenin}, var_name = {var}, ModelType = {model_type}")

def evaluate_polynomial(coefficients, soc, temp, soc_degree, temperature_degree):
    result = 0
    index = 0
    #max_degree = max(soc_degree, temperature_degree) + 1
    #comb = list(combinations_with_replacement(range(max_degree), 2))
    comb = list(product(range(soc_degree + 1), range(temperature_degree + 1)))

    for i_soc, i_temp in comb:
        if index >= len(coefficients):
            break
        result += coefficients[index] * (soc ** i_soc) * (temp ** i_temp)
        index += 1
    return result
def SocEstimation(input_path, battery_model, voltage_title, current_title, Q_counting_title, temperature_title):

    data = pd.read_csv(input_path)
    data = data[data[current_title] < 0]
    # Get the basename (filename with extension)
    filename_with_extension = os.path.basename(input_path)
    # Split the filename and extension
    filename_without_extension, _ = os.path.splitext(filename_with_extension)

    if battery_model.soc_degree == 1:
        model_dir = "Linear_estimation_" + battery_model.model_name
    else:
        model_dir = "Non_linear_estimation_" + battery_model.model_name

    if battery_model.temperature_degree != 0:
        model_dir = model_dir + "_with_temp"
        #filename_without_extension = filename_without_extension + "_temp"


    # Setting the system state variables
    A                  = battery_model.A
    B                  = -battery_model.B
    H                  = battery_model.H
    D                  = battery_model.D
    R                  = np.array([[config["rint_model_params"]["R"]]])
    coefficients       = battery_model.coefficients
    soc_degree         = battery_model.soc_degree
    temperature_degree = battery_model.temperature_degree

    if battery_model.model_type == "RintModel":
        initial_state = np.array([[config["rint_model_params"]["initial_state"]]])
        P = np.array([[config["rint_model_params"]["P"]]])
        Q = np.array([[config["rint_model_params"]["Q"]]])
    else:
        initial_state = np.array([[config["thevenin_model_params"]["initial_state"][0]], [config["thevenin_model_params"]["initial_state"][1]]])
        P = np.array([[config["thevenin_model_params"]["P"][0], 0], [0, config["thevenin_model_params"]["P"][1]]])
        Q = np.array([[config["thevenin_model_params"]["Q"][0], 0], [0, config["thevenin_model_params"]["Q"][1]]])
    delta_V = np.array([[battery_model.delta_V]])
    measurements = CsvMeasToNumpyArray(input_path, voltage_title=voltage_title, current_title=current_title, temp_title=temperature_title)  # Define vectors of [i_n, v_n]

    if battery_model.soc_degree == 1:
        filter = KalmanFilter(A=A, B=B, D=D, H=H, Q=Q, R=R, x0=initial_state, P=P, model_type=battery_model.model_type)
    else:
        filter = EKF(F=A, B=B, D=D, Q=Q, R=R, P=P, x0=initial_state, poly=battery_model.polynomial, coefficients=coefficients, soc_degree=soc_degree, temperature_degree=temperature_degree, model_type=battery_model.model_type)
    OCV_prediction = np.zeros(0)
    soc_prediction = np.zeros(0)
    os.makedirs(config["paths"]["output_directory"] + f"\{model_dir}\{filename_without_extension}", exist_ok=True)
    with open(config["paths"]["output_directory"] + f"\{model_dir}\{filename_without_extension}\logfile.txt", 'w') as file:
        file.write(f"The values of the state variables: \n A = {A}\n B = {B}\n H = {H}\n D = {D}\n Q = {Q}\n R = {R}\n soc0 = {initial_state}\n delta_V = {delta_V}\n\n")
        for i, meas in enumerate(measurements, start=0):
            if battery_model.soc_degree == 1:
                meas[1] -= delta_V
            I    = meas[0]
            V    = meas[1]
            temp = meas[2]
            #OCV_prediction.append(float(np.dot(H,  filter.predict(u = I))) + delta_V)
            filter.predict(u=I)
            x_n, p_n = filter.update(z=V, u=float(I), Temperature=temp)
            if i > 0:
                if battery_model.model_type == "RintModel":
                    if x_n > soc_prediction[i-1]:
                        x_n = (x_n + soc_prediction[i-1]) / 2
                else:
                    if x_n[1] > soc_prediction[i - 1]:
                        x_n[1] = (x_n[1] + soc_prediction[i - 1]) / 2

            if np.isnan(x_n.any()):
                soc_prediction = np.append(soc_prediction, 0)
                OCV_prediction = np.append(OCV_prediction, 0)
                print("NaN value")
            else:
                if battery_model.model_type == "RintModel":
                    soc_prediction = np.append(soc_prediction, x_n)
                    OCV_prediction = np.append(OCV_prediction, battery_model.polynomial(x_n))
                elif battery_model.temperature_degree == 0:
                    soc_prediction = np.append(soc_prediction, x_n[1])
                    OCV_prediction = np.append(OCV_prediction, battery_model.polynomial(x_n[1]))
                else:
                    OCV = evaluate_polynomial(coefficients=battery_model.coefficients, soc=x_n[1], temp=temp, soc_degree=battery_model.soc_degree, temperature_degree=battery_model.temperature_degree)
                    soc_prediction = np.append(soc_prediction, x_n[1])
                    OCV_prediction = np.append(OCV_prediction, OCV)

            line = (f"Iteration {i},  x = {x_n}, p = {p_n} \n")
            file.write(line)

    # plots
    plot_measurements(data, voltage_title, current_title, temperature_title, filename_without_extension, model_dir, battery_model.model_type, battery_model.soc_degree, battery_model.temperature_degree)
    plot_soc_time(data, Q_counting_title, soc_prediction, filename_without_extension, model_dir, battery_model.model_type, battery_model.soc_degree, battery_model.temperature_degree)
    plot_ocv_time(data, voltage_title, current_title, OCV_prediction, filename_without_extension, model_dir, float(-D), battery_model.model_type, battery_model.soc_degree, battery_model.temperature_degree)
    plot_MPE(data, soc_prediction, filename_without_extension, model_dir, battery_model.model_type, battery_model.soc_degree, battery_model.temperature_degree)
    #plot_ocv_soc(soc_prediction, OCV_prediction, filename_without_extension)

def plot_measurements(data,  voltage_title, current_title, temperature_title, filename_without_extension, model_dir, model_type, degree, temp_degree):
    if degree == 1:
        estimation_type = "KF"
    else:
        estimation_type = "EKF"
    if temp_degree != 0:
        temp = " with temperature"
    else:
        temp = ""

    OCV     = data[voltage_title]
    current = (data[current_title]) * (-1)
    time = np.linspace(0, len(OCV), len(OCV))
    temperature = data[temperature_title]

    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.plot(time, OCV, alpha=0.5, color='darkorange')
    plt.title(f'Battery Voltage [V]', fontsize=16, fontweight='bold')
    plt.xlabel('t[sec]', fontsize=14)
    plt.ylabel('Voltage[V]', fontsize=14)
    # Save the plot
    plt.savefig(config["paths"]["output_directory"] + f"\{model_dir}\{filename_without_extension}\meas_OCV_time_plot.png", dpi=300)  # Save with high resolution
    plt.close()  # Close the plot

    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.plot(time, current, alpha=0.5, color='green')
    plt.title(f'Battery Current [A]', fontsize=16, fontweight='bold')
    plt.xlabel('t[sec]', fontsize=14)
    plt.ylabel('I[A]', fontsize=14)
    # Save the plot
    plt.savefig(config["paths"]["output_directory"] + f"\{model_dir}\{filename_without_extension}\meas_current_time_plot.png", dpi=300)  # Save with high resolution
    plt.close()  # Close the plot

    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.plot(time, temperature, alpha=0.5, color='darkblue')
    plt.title(f'Battery Temperature [C]', fontsize=16, fontweight='bold')
    plt.xlabel('t[sec]', fontsize=14)
    plt.ylabel('Temperature[C]', fontsize=14)
    # Save the plot
    plt.savefig(config["paths"]["output_directory"] + f"\{model_dir}\{filename_without_extension}\meas_temperature_time_plot.png", dpi=300)  # Save with high resolution
    plt.close()  # Close the plot
def plot_soc_time(data, Q_counting_title, soc_prediction, filename_without_extension, model_dir, model_type, degree, temp_degree):
    if degree == 1:
        estimation_type = "KF"
    else:
        estimation_type = "EKF"
    if temp_degree != 0:
        temp = " with temperature"
    else:
        temp = ""
    #coulomb_count_soc = 1 - (data[Q_counting_title].values / 3000)
    commercial_soc = data[config["titles"]["SOC"]]
    time = np.linspace(0, len(soc_prediction), len(soc_prediction))

    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.plot(time, soc_prediction, label=f'{estimation_type}', alpha=0.5, color='green')
    plt.plot(time, commercial_soc, label='Commercial Estimation', alpha=0.5, color='blue')

    plt.title(f'SoC(t)', fontsize=16, fontweight='bold')
    plt.xlabel('t[sec]', fontsize=14)
    plt.ylabel('SoC', fontsize=14)
    plt.legend(loc='best', fontsize=12)
    # Save the plot
    plt.savefig(config["paths"]["output_directory"] + f"\{model_dir}\{filename_without_extension}\soc_time_plot.png", dpi=300)  # Save with high resolution
    plt.close()  # Close the plot


def plot_ocv_time(data, voltage_title, current_title, OCV_prediction, filename_without_extension, model_dir, R_0, model_type, degree, temp_degree):
    if degree == 1:
        estimation_type = "KF"
    else:
        estimation_type = "EKF"
    if temp_degree != 0:
        temp = " with temperature"
    else:
        temp = ""
    OCV            = data[voltage_title].values + abs(R_0 * data[current_title].values / 1000)
    time = np.linspace(0, OCV_prediction.shape[0], OCV_prediction.shape[0])
    OCV_prediction = OCV_prediction.reshape(time.shape)
    OCV = OCV.reshape(OCV_prediction.shape)

    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.plot(time, OCV_prediction, label=f'{estimation_type}', alpha=0.5, color='green')
    plt.plot(time, OCV, label='Measurement', alpha=0.5, color='blue')

    plt.title("$U_{oc}$(t)",  fontsize=16, fontweight='bold')

    plt.xlabel('t[sec]', fontsize=14)
    plt.ylabel('$U_{oc}$[V]', fontsize=14)
    plt.legend(loc='best', fontsize=12)
    # Save the plot
    plt.savefig(config["paths"]["output_directory"] + f"\{model_dir}\{filename_without_extension}\ocv_time_plot.png", dpi=300)  # Save with high resolution
    plt.close()  # Close the plot


def plot_MPE(data, soc_prediction, filename_without_extension, model_dir, model_type, degree, temp_degree):
    if degree == 1:
        estimation_type = "KF"
    else:
        estimation_type = "EKF"
    if temp_degree != 0:
        temp = " with temperature"
    else:
        temp = ""
    commercial_soc = data[config["titles"]["SOC"]]
    time = np.linspace(0, len(soc_prediction), len(soc_prediction))
    """
    difference = ((commercial_soc - soc_prediction)/commercial_soc) ** 2
    mpe = float(np.mean(difference))
    error = (mpe ** 0.5) * 100
    """
    difference = abs((commercial_soc - soc_prediction)) * 100
    mpe = 0
    error = np.mean(difference)

    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.plot(time, difference, label="MPE", alpha=0.5, color='green')

    plt.annotate(f'Error: {error:.2f}[%]', xy=(0.02, 0.81), xycoords='axes fraction', fontsize=12, fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.8))

    title = f'{estimation_type} {temp}'
    plt.title('Mean Percentage Error (t)', fontsize=16, fontweight='bold')
    plt.xlabel('t[sec]', fontsize=14)
    plt.ylabel('MPE[%]', fontsize=14)
    plt.legend(loc='best', fontsize=12)
    # Save the plot
    plt.savefig(config["paths"]["output_directory"] + f"\{model_dir}\{filename_without_extension}\SE_time_plot.png", dpi=300)  # Save with high resolution
    plt.close()  # Close the plot

    # Accumulate data for the combined plot

    if "Rint" not in model_dir:
        add_file_to_dict(filename_without_extension)
        error_dict[filename_without_extension]['all_times'].append(time)
        error_dict[filename_without_extension]['all_differences'].append(difference)
        error_dict[filename_without_extension]['all_labels'].append(title)
        error_dict[filename_without_extension]['all_mpes'].append(mpe)
        error_dict[filename_without_extension]['all_errors'].append(error)
        error_dict[filename_without_extension]['all_estimation_type'].append(estimation_type)
        error_dict[filename_without_extension]['all_temp'].append(temp)

def create_combined_plot():
    for filename, data in error_dict.items():
        times = data['all_times']
        differences = data['all_differences']
        labels = data['all_labels']
        mpes = data['all_mpes']
        errors = data['all_errors']
        estimation_types = data['all_estimation_type']
        temps = data['all_temp']


        plt.figure(figsize=(10, 6))
        plt.grid(True, linestyle='--', alpha=0.6)
        x_location = 0.01
        y_location = 0.97
        ystep      = 0.05

        for index, (time, difference, label, mpe, error, estimation_type, temp) in enumerate(zip(times, differences, labels, mpes, errors, estimation_types, temps)):
            # Apply Savitzky-Golay smoothing
            window_length = 101
            polyorder = 3
            difference_smooth = smooth_data_savgol(difference, window_length, polyorder)
            #plt.plot(time, difference_smooth, alpha=0.5)
            plt.plot(time, difference_smooth, label=label, alpha=0.5)
            # Annotate the plot with the MPE and error values
            max_error = np.max(difference_smooth)
            plt.annotate(f'{label}: Avg Error = {error:.3f}%, Max Error = {max_error:.3f}%', xy=(x_location, y_location), xycoords='axes fraction', fontsize=6, fontweight='bold',
                         bbox=dict(facecolor='white', alpha=0.8))
            y_location = y_location - ystep


            df = pd.read_csv(config["paths"]["output_directory"] + f"\error_comparison\error_comparison.csv", header=[0, 1], index_col=0)

            if estimation_type == "KF":
                df.at[filename, ('Max Error [%]', 'KF')] = max_error  # Under "Max Error [%]"
                df.at[filename, ('MPE [%]', 'KF')] = error  # Under "MPE [%]"
            elif estimation_type == "EKF" and temp == "":
                df.at[filename, ('Max Error [%]', 'EFK')] = max_error  # Under "Max Error [%]"
                df.at[filename, ('MPE [%]', 'EFK')] = error  # Under "MPE [%]"
            elif estimation_type == "EKF" and temp != "":
                df.at[filename, ('Max Error [%]', 'EFK with temperature')] = max_error  # Under "Max Error [%]"
                df.at[filename, ('MPE [%]', 'EFK with temperature')] = error  # Under "MPE [%]"

            df.to_csv(config["paths"]["output_directory"] + f"\error_comparison\error_comparison.csv", index=True)

        plt.title('Compare: Mean Percentage Error', fontsize=16, fontweight='bold')
        plt.xlabel('t[sec]', fontsize=14)
        plt.ylabel('MPE[%]', fontsize=14)
        plt.legend(loc='upper right', fontsize=6)
        os.makedirs(config["paths"]["output_directory"] + f"\error_comparison\{filename}", exist_ok=True)
        plt.savefig(config["paths"]["output_directory"] + f"\error_comparison\{filename}\combined_errors_plot", dpi=300)
        plt.close()  # Close the plot

def smooth_data_savgol(data, window_length, polyorder):
    return savgol_filter(data, window_length, polyorder)

def add_file_to_dict(filename):
    if filename not in error_dict:
        # Initialize the lists if the key does not exist
        error_dict[filename] = {
            'all_times': [],
            'all_differences': [],
            'all_labels': [],
            'all_mpes': [],
            'all_errors': [],
            'all_estimation_type': [],
            'all_temp': []
        }


if __name__ == '__main__':

    with open('config.json', 'r') as file:
        config = json.load(file)

    # Global Params for errors combined plot
    error_dict = {}
    models_list = []
    for i in config["paths"]["model"]:
        model_name = SliceFileName(i)
        globals()[f"rint_linear_{model_name}"] = CreateBatteryModel(
            input_path=i,
            r0=config["rint_model_params"]["r0"],
            q=config["rint_model_params"]["q"],
            s=config["rint_model_params"]["s"],
            p=config["rint_model_params"]["p"],
            voltage_title=config["titles"]["voltage"],
            current_title=config["titles"]["current"],
            Q_counting_title=config["titles"]["q_counting"],
            Time_title=config["titles"]["time"],
            Temperature_title=config["titles"]["temperature"],
            soc_title=config["titles"]["SOC"],
            Temperature_flag=config["rint_model_params"]["temperature_flag"],
            AdjustedSoc=config["rint_model_params"]["adjusted_soc"],
            model=ModelType.RINT,
            soc_degree=config["rint_model_params"]["soc_degree"][0],
            temperature_degree=config["rint_model_params"]["temperature_degree"]
        )
        models_list.append(globals()[f"rint_linear_{model_name}"])

        globals()[f"rint_nonlinear_{model_name}"] = CreateBatteryModel(
            input_path=i,
            r0=config["rint_model_params"]["r0"],
            q=config["rint_model_params"]["q"],
            s=config["rint_model_params"]["s"],
            p=config["rint_model_params"]["p"],
            voltage_title=config["titles"]["voltage"],
            current_title=config["titles"]["current"],
            Q_counting_title=config["titles"]["q_counting"],
            Time_title=config["titles"]["time"],
            Temperature_title=config["titles"]["temperature"],
            soc_title=config["titles"]["SOC"],
            Temperature_flag=config["rint_model_params"]["temperature_flag"],
            AdjustedSoc=config["rint_model_params"]["adjusted_soc"],
            model=ModelType.RINT,
            soc_degree=config["rint_model_params"]["soc_degree"][1],
            temperature_degree=config["rint_model_params"]["temperature_degree"]
        )
        models_list.append(globals()[f"rint_nonlinear_{model_name}"])

        globals()[f"thevenin_linear_{model_name}"] = CreateBatteryModel(
            input_path=i,
            r0=config["thevenin_model_params"]["r0"],
            r1=config["thevenin_model_params"]["r1"],
            c1=config["thevenin_model_params"]["c1"],
            q=config["thevenin_model_params"]["q"],
            s=config["thevenin_model_params"]["s"],
            p=config["thevenin_model_params"]["p"],
            voltage_title=config["titles"]["voltage"],
            current_title=config["titles"]["current"],
            Q_counting_title=config["titles"]["q_counting"],
            Time_title=config["titles"]["time"],
            Temperature_title=config["titles"]["temperature"],
            soc_title=config["titles"]["SOC"],
            Temperature_flag=config["thevenin_model_params"]["temperature_flag"][0],
            AdjustedSoc=config["thevenin_model_params"]["adjusted_soc"],
            model=ModelType.THEVENIN,
            soc_degree=config["thevenin_model_params"]["soc_degree"][0],
            temperature_degree=config["thevenin_model_params"]["temperature_degree"][0]
        )
        models_list.append(globals()[f"thevenin_linear_{model_name}"])

        globals()[f"thevenin_nonlinear_{model_name}"] = CreateBatteryModel(
            input_path=i,
            r0=config["thevenin_model_params"]["r0"],
            r1=config["thevenin_model_params"]["r1"],
            c1=config["thevenin_model_params"]["c1"],
            q=config["thevenin_model_params"]["q"],
            s=config["thevenin_model_params"]["s"],
            p=config["thevenin_model_params"]["p"],
            voltage_title=config["titles"]["voltage"],
            current_title=config["titles"]["current"],
            Q_counting_title=config["titles"]["q_counting"],
            Time_title=config["titles"]["time"],
            Temperature_title=config["titles"]["temperature"],
            soc_title=config["titles"]["SOC"],
            Temperature_flag=config["thevenin_model_params"]["temperature_flag"][0],
            AdjustedSoc=config["thevenin_model_params"]["adjusted_soc"],
            model=ModelType.THEVENIN,
            soc_degree=config["thevenin_model_params"]["soc_degree"][1],
            temperature_degree=config["thevenin_model_params"]["temperature_degree"][0]
        )
        models_list.append(globals()[f"thevenin_nonlinear_{model_name}"])

        globals()[f"thevenin_nonlinear_with_temp_{model_name}"] = CreateBatteryModel(
            input_path=i,
            r0=config["thevenin_model_params"]["r0"],
            r1=config["thevenin_model_params"]["r1"],
            c1=config["thevenin_model_params"]["c1"],
            q=config["thevenin_model_params"]["q"],
            s=config["thevenin_model_params"]["s"],
            p=config["thevenin_model_params"]["p"],
            voltage_title=config["titles"]["voltage"],
            current_title=config["titles"]["current"],
            Q_counting_title=config["titles"]["q_counting"],
            Time_title=config["titles"]["time"],
            Temperature_title=config["titles"]["temperature"],
            soc_title=config["titles"]["SOC"],
            Temperature_flag=config["thevenin_model_params"]["temperature_flag"][1],
            AdjustedSoc=config["thevenin_model_params"]["adjusted_soc"],
            model=ModelType.THEVENIN,
            soc_degree=config["thevenin_model_params"]["soc_degree"][1],
            temperature_degree=config["thevenin_model_params"]["temperature_degree"][1]
        )
        models_list.append(globals()[f"thevenin_nonlinear_with_temp_{model_name}"])


    for index, input_file in enumerate(config["paths"]["input_files"]):
        for model in models_list:
            SocEstimation(input_path=input_file,
                          battery_model=model,
                          voltage_title=config["titles"]["voltage"],
                          current_title=config["titles"]["current"],
                          Q_counting_title=config["titles"]["q_counting"],
                          temperature_title=config["titles"]["temperature"]
                          )

        if index == 0:
            columns = pd.MultiIndex.from_tuples([
                ('MPE [%]', 'KF'),
                ('MPE [%]', 'EFK'),
                ('MPE [%]', 'EFK with temperature'),
                ('Max Error [%]', 'KF'),
                ('Max Error [%]', 'EFK'),
                ('Max Error [%]', 'EFK with temperature')
            ])
            df = pd.DataFrame(columns=columns)
            os.makedirs(config["paths"]["output_directory"] + f"\error_comparison", exist_ok=True)
            df.to_csv(config["paths"]["output_directory"] + f"\error_comparison\error_comparison.csv")
    create_combined_plot()

