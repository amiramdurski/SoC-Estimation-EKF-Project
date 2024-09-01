import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from enum import IntEnum
from BatteryModel import BatteryModel, RintModel, TheveninModel
import chardet

class ModelType(IntEnum):
    RINT     = 1
    THEVENIN = 2

def CsvMeasToNumpyArray(file_path, voltage_title, current_title, temp_title):
    measurements = np.zeros((0,3))
    df = pd.read_csv(file_path)
    df = df[df[current_title] < 0]
    for voltage, current, temp in zip(df[voltage_title], df[current_title], df[temp_title]):
        new_meas = np.array([abs(current)/1000, abs(voltage), abs(temp)])
        measurements = np.vstack((measurements, new_meas))
    return measurements


def CreateSocVector(data, Q_counting_title, output_path):
    # Create a new column 'AdjustedRemPct' and initialize
    data['AdjustedRemPct'] = 0.0
    # Initialize a new dictionary with value and proper decrement
    dict = {}
    # Add to dictionary: key = original value of RemPct, value = [count of same value, proper decrement]
    for value in data[Q_counting_title].unique():
        # Count occurrences of the value
        count = data[data[Q_counting_title] == value].shape[0]
        dict[value] = [count, 1 / count]
    # Create list with all the new RemPct values
    new_column = []
    for key, value in dict.items():
        for i in range(value[0]):
            new_column.append((key + 1 - i * value[1]) / 100)
    # Add the new RemPct column to the data frame
    data['AdjustedRemPct'] = new_column
    # Save to a new CSV file
    data.to_csv(output_path, index=False)


def SliceFileName(input_path):
    # Get the basename (filename with extension)
    filename_with_extension = os.path.basename(input_path)
    # Split the filename and extension
    filename_without_extension, _ = os.path.splitext(filename_with_extension)
    return filename_without_extension


def CreateBatteryModel(input_path, r0, q, s, p, voltage_title, current_title, Q_counting_title, Time_title, Temperature_title, soc_title, Temperature_flag, AdjustedSoc = False, model=ModelType.RINT, soc_degree=1, temperature_degree=0, r1=0, c1=0):

    filename_without_extension = SliceFileName(input_path)

    # Load the CSV file
    data = pd.read_csv(input_path)

    if model == ModelType.RINT:
        battery_model = RintModel(data, q, s, p, soc_degree, temperature_degree, r0, Temperature_flag, filename_without_extension)
    elif model == ModelType.THEVENIN:
        battery_model = TheveninModel(data, q, s, p, soc_degree, temperature_degree, r0, Temperature_flag, filename_without_extension, r1, c1)

    battery_model.CalcSoC(AdjustedSoc, Q_counting_title, soc_title)
    battery_model.CalcOCV(voltage_title, current_title)
    battery_model.CreateTemperatureVec(Temperature_title)
    battery_model.FitPoly()
    battery_model.PrintModel(filename_without_extension)
    battery_model.CalcDeltaT(Time_title)
    battery_model.CalcModelParameters()

    return (battery_model)


if __name__ == "__main__":
    input_path = 'C:\python\FinalProject\output.csv'  # Path to your input CSV file
    #output_path = 'C:\python\FinalProject\outputs\gen_output.csv'  # Path to save the modified CSV file
    #CsvMeasToNumpyArray(input_path)
    #CreateBatteryModel(input_path = input_path, output_path = output_path, r0 = 50e-3, q = 78, s = 6, p = 4)
