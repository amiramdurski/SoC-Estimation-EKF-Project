import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D



# Defining Global Constant
CURRENT_UNITS = 1000  # It is 1/units



class BatteryModel:
    def __init__(self, data, q, s, p, soc_degree, temperature_degree, r0, Temperature_flag, filename_without_extension):
        self.data = data
        self.q = q
        self.s = s
        self.p = p
        self.r0 = r0
        self.Q  = self.q * self.p * self.s
        self.R0 = (self.r0 * self.s) / self.p
        self.soc_degree = soc_degree
        self.temperature_degree = temperature_degree
        self.model_name = filename_without_extension
        self.Temperature_flag = Temperature_flag

    def CalcOCV(self):
        raise NotImplementedError("Method CalcOCV not implemented")

    def CalcSoC(self, AdjustedSoc, Q_counting_title, soc_title):
        if AdjustedSoc == True:
            self.SoC = self.data[soc_title].values
        else:
            self.SoC = 1 - (self.data[Q_counting_title].values / self.Q) # Need to be more robust for diffrent csv
            # AdjustedRemPct = (data[Q_counting_title].values - data[Q_counting_title][0]) / data[Q_counting_title].iloc[-1]
            # AdjustedRemPct = 1 - (AdjustedRemPct / AdjustedRemPct[-1])

    def CreateTemperatureVec(self, Temperature_title):
        self.Temperature = self.data[Temperature_title]
    def CalcDeltaT(self, Time_title):
        # Calculation of delta t average
        differences = self.data[Time_title].diff()
        self.delta_t = differences.mean()

    def FitPoly(self):
        self.polynomial = 0
        if self.Temperature_flag == False:
            self.coefficients = np.polyfit(self.SoC, self.OCV, self.soc_degree)
            # Create a polynomial function from the coefficients
            self.polynomial = np.poly1d(self.coefficients)
        else: #self.Temperature_flag == True
            # Combine soc and temp into a single matrix
            X = np.column_stack((self.SoC, self.Temperature))
            y = self.OCV

            # Generate the polynomial features
            soc_features = np.array([self.SoC ** i for i in range(self.soc_degree + 1)]).T
            temp_features = np.array([self.Temperature ** j for j in range(self.temperature_degree + 1)]).T
            poly_features = np.hstack(
                [soc_features[:, i:i + 1] * temp_features[:, j:j + 1] for i in range(self.soc_degree + 1) for j in
                 range(self.temperature_degree + 1)])

            # Fit the model
            self.coefficients, _, _, _ = np.linalg.lstsq(poly_features, y, rcond=None)

            # Print Equation of the Model
            equation = "V_OC = "
            terms = []
            index = 0
            for i in range(self.soc_degree + 1):
                for j in range(self.temperature_degree + 1 ):
                    if self.coefficients[index] != 0:
                        term = f"{self.coefficients[index]:.4f}*soc^{i}*temp^{j}"
                        terms.append(term)
                    index += 1
            equation += " + ".join(terms)
            print(equation)

    def PrintModel(self, filename):
        self.filename = filename

        if self.Temperature_flag == False:
            # Generate y-values based on the fitted polynomial
            x_fit = np.linspace(min(self.SoC), max(self.SoC), 100)
            y_fit = self.polynomial(x_fit)

            plt.figure(figsize=(10, 6))
            # Plotting the original data points
            #plt.plot(self.SoC, self.OCV, color='blue', label='Measurement', alpha=0.5, linewidth=2, linestyle='--')
            plt.plot(self.SoC, self.OCV, color='blue', label='Measurement', alpha=0.5, markersize=0.5, linestyle='None', marker='.')

            # Plotting the fitted polynomial curve and add title, labels, and legend
            if self.soc_degree == 1:
                plt.plot(x_fit, y_fit, color='purple',  label='Linear Model', linewidth=3, linestyle='-')
                plt.title('$U_{oc}$(SoC)', fontsize=16, fontweight='bold')
            else:
                plt.plot(x_fit, y_fit, color='purple',  label='Polynomial Model', linewidth=3, linestyle='-')
                plt.title('$U_{oc}$(SoC)', fontsize=16, fontweight='bold')
            plt.xlabel('SoC', fontsize=14)
            plt.ylabel('$U_{oc}$[V]', fontsize=14)
            plt.legend(loc='best', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)

            # Enhance tick parameters
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)

            # Check the directory exists
            os.makedirs(f"C:\python\FinalProject\outputs\{self.model_name}\{self.filename}", exist_ok=True)
            # Save the plot
            plt.savefig(f"C:\python\FinalProject\outputs\{self.model_name}\{self.filename}\OCV_Soc.png", dpi=300)  # Save with high resolution
            plt.close()  # Close the plot
        else: #self.Temperature_flag == True
            # self.SoC, self.Temperature, self.OCV
            soc_range = np.linspace(self.SoC.max(), self.SoC.min(), 50)
            temp_range = np.linspace(self.Temperature.min(), self.Temperature.max(), 50)
            soc_grid, temp_grid = np.meshgrid(soc_range, temp_range)

            # Generate the polynomial features for the grid
            soc_grid_features = np.array([soc_grid.ravel() ** i for i in range(self.soc_degree + 1)]).T
            temp_grid_features = np.array([temp_grid.ravel() ** j for j in range(self.temperature_degree + 1)]).T
            poly_grid_features = np.hstack(
                [soc_grid_features[:, i:i + 1] * temp_grid_features[:, j:j + 1] for i in range(self.soc_degree + 1) for j in
                 range(self.temperature_degree + 1)])

            v_oc_grid = poly_grid_features @ self.coefficients
            v_oc_grid = v_oc_grid.reshape(soc_grid.shape)

            plt.figure(figsize=(14, 8))

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.SoC, self.Temperature, self.OCV, c='b', marker='o', s=2, alpha=0.1)
            surf = ax.plot_surface(soc_grid, temp_grid, v_oc_grid, cmap='viridis', edgecolor='k', rstride=5, cstride=5, alpha=0.5)
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, pad=0.1)
            ax.set_xlabel('SoC', fontsize=12, labelpad=10)
            ax.set_ylabel('Temperature[C]', fontsize=12, labelpad=10)
            ax.set_zlabel('$U_{oc}$[V]', fontsize=12, labelpad=10)
            ax.invert_xaxis()  # Invert the SOC axis
            # Enhance tick parameters
            ax.tick_params(axis='x', labelsize=10)
            ax.tick_params(axis='y', labelsize=10)
            ax.tick_params(axis='z', labelsize=10)

            # Check the directory exists
            os.makedirs(f"C:\python\FinalProject\outputs\{self.model_name}\{self.filename}", exist_ok=True)
            # Save the plot
            plt.savefig(f"C:\python\FinalProject\outputs\{self.model_name}\{self.filename}\OCV_Soc.png", dpi=300)  # Save with high resolution
            plt.close()  # Close the plot

        """
        # Write to Log File
        with open(f"C:\python\FinalProject\outputs\{self.model_name}\{self.filename}\logfile.txt", 'w') as file:
            file.write(f"The values of the physical model:\n")
            file.write(
                f"Q = {self.Q}\n R_0 = {self.R0}\n A = {float(self.A)}\n B = {float(self.B)}\n H = {float(self.H)}\n D = {float(self.D)}\n delta_V = {float(self.delta_V)}\n\n")
        """


    def CalcModelParameters(self):
        raise NotImplementedError("Method CalcModelParameters not implemented")

class RintModel(BatteryModel):
    def __init__(self, data, q, s, p, soc_degree, temperature_degree, r0, Temperature_flag, filename_without_extension):
        super().__init__(data, q, s, p, soc_degree, temperature_degree, r0, Temperature_flag, filename_without_extension)
        self.model_name += "_RintModel_" + str(self.soc_degree) + "_degree"
        if self.Temperature_flag == True:
            self.model_name += "_temp"
        self.model_type = "RintModel"

    def CalcOCV(self, voltage_title, current_title):
        self.OCV = self.data[voltage_title].values + abs(self.R0 * self.data[current_title].values / CURRENT_UNITS)

    def CalcModelParameters(self):
        self.A       = np.array([[1]])
        self.B       = np.array([[(self.delta_t) / (3600 * self.Q)]])
        self.H       = np.array([[self.coefficients[0]]])
        self.D       = np.array([-self.R0])
        self.delta_V = self.coefficients[1]


class TheveninModel(BatteryModel):
    def __init__(self, data, q, s, p, soc_degree, temperature_degree, r0, Temperature_flag, filename_without_extension, r1, c1):
        super().__init__(data, q, s, p, soc_degree, temperature_degree, r0, Temperature_flag, filename_without_extension)
        self.r1 = r1
        self.c1 = c1
        self.R1 = (self.r1 * self.s) / self.p
        self.C1 = (self.c1 * self.p) / self.s
        self.model_name += "_TheveninModel_" + str(self.soc_degree) + "_degree"
        if self.Temperature_flag == True:
            self.model_name += "_temp"
        self.model_type = "TheveninModel"


    def CalcOCV(self, voltage_title, current_title):
        self.OCV = self.data[voltage_title].values + abs(self.R0 * self.data[current_title].values / CURRENT_UNITS)

    def CalcModelParameters(self):
        e            = np.exp((-self.delta_t) / (3600 * self.c1 * self.R1))
        self.A       = np.array([[e, 0], [0, 1]])
        self.B       = np.array([[self.r1 * (1 - e)], [(self.delta_t) / (3600 * self.Q)]])
        self.H       = np.array([[-1, self.coefficients[0]]])
        self.D       = np.array([-self.R0])
        self.delta_V = self.coefficients[1]


"""
    def MultidimensionalPolynomial(self):
        result = 0
        index = 0
        max_degree = max(self.soc_degree, self.temperature_degree) + 1
        comb = list(combinations_with_replacement(range(max_degree), 2))
        for i_soc, i_temp in comb:
            if index >= len(self.coefficients):
                break
            result += self.coefficients[index] * (self.SoC ** i_soc) * (self.Temperature ** i_temp)
            index += 1
        return result

    def DerivativeSoc(self):
        result = 0
        index = 0
        max_degree = max(self.soc_degree, self.temperature_degree) + 1
        comb = list(combinations_with_replacement(range(max_degree), 2))
        for i_soc, i_temp in comb:
            if index >= len(self.coefficients):
                break
            if i_soc > 0:
                result += self.coefficients[index] * i_soc * (self.SoC ** (i_soc - 1)) * (self.Temperature ** i_temp)
            index += 1
        return result

"""