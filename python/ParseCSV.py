import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def csv_measurements_to_numpy_array(file_path):
    measurements = np.zeros((0,2))
    df = pd.read_csv(file_path)
    for voltage, current in zip(df['Volt'], df['Curr']):
        new_meas = np.array([current, voltage])
        measurements = np.vstack((measurements, new_meas))
    return measurements


def creating_physical_model(input_path, output_path, r_0, Q, s, p):
    # Load the CSV file
    data = pd.read_csv(input_path)

    # Create a new column 'AdjustedRemPct' and initialize
    data['AdjustedRemPct'] = 0.0

    # Initialize a new dictionary with value and proper decrement
    dict = {}

    # Add to dictionary: key = original value of RemPct, value = [count of same value, proper decrement]
    for value in data['RemPct'].unique():
        # Count occurrences of the value
        count = data[data['RemPct'] == value].shape[0]
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

    # create graph of Voltage as function of Adjusted RemPct and linear regression trendline
    # Calculate linear regression (ordinary least squares)

    AdjustedRemPct = data['AdjustedRemPct'].values
    Voltage        = data['VoltR'].values
    X = np.vstack([AdjustedRemPct, np.ones(len(AdjustedRemPct))]).T
    m, c = np.linalg.lstsq(X, Voltage, rcond=None)[0]

    # Plot regular and linear trendline
    plt.plot(AdjustedRemPct, Voltage, label='Regular Trendline', alpha=0.5)
    plt.plot(AdjustedRemPct, m * AdjustedRemPct + c, 'r-', label='Linear Trendline')

    # Add equation of the linear trendline
    equation_text = f'y = {m:.2f}x + {c:.2f}'
    plt.text(0.5, 0.9, equation_text, fontsize=12, transform=plt.gca().transAxes)

    # Add title, labels, and legend
    plt.title('Voltage as function of SoC')
    plt.xlabel('SOC')
    plt.ylabel('Voltage Measurment [V]')
    plt.legend()
    plt.show()

    # Calculation of internal resistance
    R_0 = (r_0 * s) / p

    # Calculation of delta t average
    differences = data['Time'].diff()
    delta_t = differences.mean()

    A = 1
    B = (delta_t * 10**(-6)) / (3600 * Q)
    H = m
    D = -R_0
    delta_V = c
    return (A, B, H, D, delta_V)


if __name__ == "__main__":
    input_path = 'C:\python\FinalProject\output.csv'  # Path to your input CSV file
    output_path = 'C:\python\FinalProject\gen_output.csv'  # Path to save the modified CSV file
    #csv_measurements_to_numpy_array(input_path)
    #creating_physical_model(input_path = input_path, output_path = output_path, r_0 = 50e-3, Q = 78, s = 6, p = 4)
