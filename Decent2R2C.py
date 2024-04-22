#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 15:54:29 2023

@author: ozanbaris
"""

from scipy.optimize import differential_evolution, minimize
import numpy as np
from scipy.linalg import expm, inv
from scipy.optimize import differential_evolution, minimize
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd
class R2C2Decent:
    def __init__(self, Ri, Re, Ci, Ce, Ai, Ae, roxP_hvac):
        """
        Initialize the 1R1C model with the given parameters.
        
        Parameters:
        R (float): Thermal resistance (K/W).
        C(float): Thermal capacity (J/K).
        A (float): Solar aperture coefficient.
        roxP_hvac (float): Power input of the HVAC system.
        """
        # Model parameters
        self.Ri = Ri # The first resistance (K/W) of the model, on the indoor side
        self.Re = Re # The second resistance (K/W) of the model, on the outdoor side
        self.Ci = Ci # Heat capacity (J/K) connected to the indoor temperature
        self.Ce = Ce # Heat capacity (J/K) connected to the envelope temperature
        self.Ai = Ai # Solar aperture coefficient directed to the indoor temperature
        self.Ae = Ae # Solar aperture coefficient directed to the envelope temperature
        self.roxP_hvac = roxP_hvac
        self.N_states =2
        
        self.update_matrices()
    def update_matrices(self):
        
        # Matrices of the continuous system
        self.Ac = np.array([[-1/(self.Ci*self.Ri), 1/(self.Ci*self.Ri)],
                            [1/(self.Ce*self.Ri), -1/(self.Ce*self.Ri) - 1/(self.Ce*self.Re)]])
        self.Bc = np.array([[0, -self.roxP_hvac/ self.Ci, self.Ai / self.Ci],
                            [1/(self.Ce*self.Re), 0, self.Ae/self.Ce]])
        
        self.Cc = np.array([[1, 0]])
    
    def discretize(self, dt):
        """ This method applies the discretisation equations shown above"""
        n = self.N_states
        # Discretisation
        F = expm(self.Ac * dt)
        G = np.dot(inv(self.Ac), np.dot(F - np.eye(n), self.Bc))
        H = self.Cc

        return F, G

    def predict(self, T,Te, T_ext, u, ghi,  dt):
        """
        Update the indoor temperatures for the next time step using the Euler method in a vectorized form.
        
        Parameters:
        T (np.array): indoor temperatures (K)  for sensor.
        Te (np.array): envelope temperatures (K) for sensor.
        T_ext (np.array): External temperatures (K) .
        u (np.array): Duration HVAC was on (duty cycle) 
        ghi (np.array): Solar radiation (W/m2) .
        delta_t (float): Time step (sec).
        
        Returns:
        np.array: Temperatures at the next time step (F) for each time step or sensor.
        """

        F, G = self.discretize(dt)
        
        # Stack T and Te vertically, then transpose to get a 2xN matrix where N is the number of samples
        state_matrix = np.vstack((T, Te)).T  # Each row is [T, Te] for a timestep
        
        # Similarly, stack T_ext, u, and ghi vertically and transpose
        input_matrix = np.vstack((T_ext, u, ghi)).T  # Each row is [T_ext, u, ghi] for a timestep
        
        # Perform the matrix multiplication in a batch
        predictions = (F @ state_matrix.T) + (G @ input_matrix.T)
        
        # Transpose the result back so each row corresponds to a timestep
        predictions = predictions.T
        
        # And only return the first column, which represents T
        return predictions[:, 0]


    def evaluate(self, parameters, sensor, df):
        """
        Evaluates the model by making predictions and comparing them to observed data.

        Parameters:
        - parameters: Dictionary containing model parameters R, C, A, roxP_hvac, delta_t.
        - df: DataFrame with observed data, including control inputs (u, ghi, T_ext) and temperatures.

        Returns:
        - RMSE: Root Mean Square Error between the observed and predicted temperatures.
        """
        # Unpack parameters
        dt = 3600 #since data is in hours

        # Extract inputs from DataFrame
        T = df[sensor].values[:-1]  # Initial temperatures for prediction
        T_ext = df['Text'].values[:-1]  # External temperatures
        Te=(T+T_ext)/2
        u = df['duty_cycle'].values[:-1]  # HVAC duty cycle
        ghi = df['GHI'].values[:-1]  # Solar radiation
        
        # Predict the next temperatures
        predictions = self.predict(T, Te, T_ext, u, ghi, dt)
        
        # Observed temperatures for comparison (excluding the first value used as initial condition)
        y_obs = df[sensor].values[1:]
        
        # Calculate RMSE between observed temperatures and predictions
        rmse = np.sqrt(mean_squared_error(y_obs, predictions))
        
        return rmse
    def set_parameters(self, params):
        """
        Sets the parameters of the model based on the input dictionary.
        """

        self.Ri = params['Ri'] # The first resistance (K/W) of the model, on the indoor side
        self.Re = params['Re'] # The second resistance (K/W) of the model, on the outdoor side
        self.Ci = params['Ci'] # Heat capacity (J/K) connected to the indoor temperature
        self.Ce = params['Ce'] # Heat capacity (J/K) connected to the envelope temperature
        self.Ai = params['Ai'] # Solar aperture coefficient directed to the indoor temperature
        self.Ae = params['Ae'] # Solar aperture coefficient directed to the envelope temperature
        self.roxP_hvac = params['roxP_hvac']
        
        self.update_matrices()

        

#%%

def fitness_function(params, model, df, sensor):
    """
    Fitness function for the optimization algorithm.
    It evaluates the model with the given parameters against the observed data
    and returns the negative RMSE as the fitness value to be minimized.
    """
    param_dict = {
    'Ri': params[0],
    'Re': params[1],
    'Ci': params[2],
    'Ce': params[3],
    'Ai': params[4],
    'Ae': params[5],
    'roxP_hvac': params[6]
    }
    
    # Set model parameters
    model.set_parameters(param_dict)
    
    # Assuming 'sensor' column in df holds the sensor temperature data
    # and df is properly structured with 'duty_cycle', 'GHI', and 'Text' columns
    try:
        rmse = model.evaluate(params, sensor, df)
        if np.isnan(rmse) or np.isinf(rmse):
            return np.finfo(np.float64).max
        return rmse
    except OverflowError as e:
        print(f"Overflow with parameters {params}: {e}")
        return np.finfo(np.float64).max
    


def optimize_parameters(model, df, sensor, bounds):
    """
    Optimizes the parameters of the model.
    This function attempts to optimize parameters using differential evolution (DE) first.
    If DE is successful, it proceeds to refine the optimization with Sequential Least Squares Programming (SLSQP).
    If any step fails due to a ValueError, indicating issues with the input, it will report the failure.
    """
    def fitness_wrapper(params):
        # This function should handle exceptions or ensure that params do not cause errors in the model.
        try:
            return fitness_function(params, model, df, sensor)
        except ValueError as e:
            print(f"Error in fitness function for sensor {sensor}: {e}")
            # Return a high cost to indicate error without stopping the optimization process
            return float('inf')

    try:
        # Attempt global optimization with DE
        result_ga = differential_evolution(fitness_wrapper, bounds, strategy='best1bin',
                                           maxiter=100, popsize=15, tol=0.01,
                                           mutation=(0.2, 1.8), recombination=0.7, disp=True)

        print(f"Optimized RMSE after GA: {result_ga.fun}")
        initial_guess = result_ga.x

    except ValueError as e:
        print(f"Error encountered during DE optimization for sensor {sensor}: {e}")
        # Provide a default initial guess if DE fails
        initial_guess = [0.5 * (b[0] + b[1]) for b in bounds]

    try:
        # Proceed to SLSQP optimization
        result_slsqp = minimize(fitness_wrapper, initial_guess, method='SLSQP', bounds=bounds,
                                options={'maxiter': 100, 'disp': True})

        print(f"Refined Optimized RMSE: {result_slsqp.fun}")

        optimized_dict = {
            'Ri': result_slsqp.x[0],
            'Re': result_slsqp.x[1],
            'Ci': result_slsqp.x[2],
            'Ce': result_slsqp.x[3],
            'Ai': result_slsqp.x[4],
            'Ae': result_slsqp.x[5],
            'roxP_hvac': result_slsqp.x[6],
        }

        return optimized_dict, result_slsqp.fun
    except ValueError as e:
        print(f"Optimization failed during SLSQP for sensor {sensor}: {e}")
        # Return None or some indicator that optimization failed
        return None, None




#%%

def set_rc_model_from_distribution(sensor_count, bounds):


    # Selecting parameters from the uniform distribution within the specified bounds
    Ri= np.random.uniform(bounds[0][0], bounds[0][1])
    Re= np.random.uniform(bounds[1][0], bounds[1][1])
    Ci = np.random.uniform(bounds[2][0], bounds[2][1])
    Ce = np.random.uniform(bounds[3][0], bounds[3][1])
    Ai = np.random.uniform(bounds[4][0], bounds[4][1])
    Ae = np.random.uniform(bounds[5][0], bounds[5][1])
    roxP_hvac = np.random.uniform(bounds[6][0], bounds[6][1])

    # Creating an RCModel instance with the randomly selected parameters
    rc_model = R2C2Decent(Ri, Re,Ci, Ce, Ai, Ae, roxP_hvac)
    
    return rc_model

def fit_models_R2C2Decent(processed_houses_reduced):
    # Dictionary to store optimization results for each house
    optimization_results_R2C2_decent = {}
        # Start timing
    start_time = time.time()
    for add_sensor_count, houses in processed_houses_reduced.items():
        optimization_results_R2C2_decent[add_sensor_count] = {}
        for house_id, df in houses.items():
            sensor_count = add_sensor_count + 1
            print('sensor count', sensor_count, 'house id', house_id)

            test_size = 0.125
            num_test_samples = int(len(df) * test_size)

            # Calculate the split index
            split_index = len(df) - num_test_samples

            # Split the DataFrame without shuffling
            train_df = df.iloc[:split_index]
            test_df = df.iloc[split_index:]
            
            aggregated_params = {'Ri': [], 'Re': [], 'Ci': [], 'Ce': [], 'Ai': [], 'Ae': [], 'roxP_hvac': []}
            rmse_test_collect = []
            rmse_train_collect = []
            for sensor_index in range(1, sensor_count + 1):
                sensor_column = f"T0{sensor_index}_TEMP"

                bounds = [
                    (0.00001, 0.05),  # Ri bounds
                    (0.00001, 0.1),  # Re bounds
                    (10000, 10000000),  # Ci bounds
                    (10000, 10000000),  # Ce bounds
                    (0.00001, 1.5),  # Ai bound
                    (0.00001, 1.5),  # Ae bound
                    (200, 10000)  # roxP_hvac bound
                ]

                rc_model = set_rc_model_from_distribution(sensor_count, bounds)
                
                optimal_params, rmse_train = optimize_parameters(rc_model, train_df, sensor_column, bounds)
                
                if optimal_params is not None:
                    for param in ['Ri', 'Re', 'Ci', 'Ce', 'Ai', 'Ae', 'roxP_hvac']:
                        aggregated_params[param].append(optimal_params[param])
                    
                    rc_model.set_parameters(optimal_params)
                    rmse_test = rc_model.evaluate(optimal_params, sensor_column, test_df)
                    
                    rmse_test_collect.append(rmse_test)
                    rmse_train_collect.append(rmse_train)
                else:
                    print(f"Optimization failed for sensor {sensor_column} in house {house_id}. Skipping this sensor.")
                    continue

            optimization_results_R2C2_decent[add_sensor_count][house_id] = {
                'optimal_params': aggregated_params,
                'rmse_train': np.mean(rmse_train_collect),
                'rmse_test': np.mean(rmse_test_collect)
            }
    
        # End timing
    end_time = time.time()

    # Calculate and print the execution time
    execution_time_minutes = (end_time - start_time) / 60
    print(f"The loop took {execution_time_minutes:.2f} minutes to run.")

    return optimization_results_R2C2_decent

               
#%%

def plot_parameter_distributions_R2C2Decent(optimization_results):
    parameters = ['Ri', 'Re', 'Ci', 'Ai', 'roxP_hvac']
    parameter_titles = ['$R_{i}$ Values Distribution (K/W)', '$R_{e}$ Values Distribution (K/W)', 
                        '$C_{i}$ Values Distribution (J/K)', '$\\alpha_{i}$ Values Distribution $(m^2)$', 
                        '$\\rho \\varphi_{hvac}$ Distribution (W)']
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    plt.rcParams['font.size'] = 20
    
    fig, axs = plt.subplots(5, 1, figsize=(10, 20))
    sensor_counts = sorted(optimization_results.keys())

    for param_index, param in enumerate(parameters):
        for sensor_idx, sensor_count in enumerate(sensor_counts):
            all_values = []
            houses = optimization_results[sensor_count]
            for house_id, results in houses.items():
                if results['rmse_test'] <= 0.5:
                    param_key = param if param != 'roxP_hvac' else 'roxP_hvac'
                    all_values.extend(results['optimal_params'][param_key])
            
            if not all_values:
                continue

            sns.histplot(all_values, kde=False, ax=axs[param_index], stat="density", 
                         label=f"Sensor {sensor_count}", color=colors[sensor_idx % len(colors)], 
                         bins=100, alpha=0.5, edgecolor='black')
            sns.kdeplot(all_values, ax=axs[param_index], color=colors[sensor_idx % len(colors)])

            axs[param_index].set_xlim([min(all_values), max(all_values)])

        axs[param_index].set_xlabel(parameter_titles[param_index], fontsize=18)
        axs[param_index].set_ylabel('Density', fontsize=16)
        
        if param_index == len(parameters) - 1:
            axs[param_index].legend(title="Sensor Count", ncol=2)
        else:
            axs[param_index].legend([],[], frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)  # Adjust the vertical space between plots here
    #plt.savefig('parameter_distribution.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.show()



def save_results_R2C2Decent(optimization_results_decent):
    """
    Saves optimization results to CSV, with a separate file for each sensor count.

    Parameters:
    - optimization_results_decent: A dictionary with nested structure [sensor_count][house_id] containing optimization outcomes.
    """
    for sensor_count, houses in optimization_results_decent.items():
        # Initialize lists to store the data
        data = {
            'House ID': [],
            'RMSE Train': [],
            'RMSE Test': [],
            'Ri_values': [],
            'Ci_values': [],
            'Ai_values': [],
            'Re_values': [],
            'Ce_values': [],
            'Ae_values': [],
            'roxP_hvac': []
        }

        # Populate the data structure
        for house_id, results in houses.items():
            data['House ID'].append(house_id)
            data['RMSE Train'].append(results['rmse_train'])
            data['RMSE Test'].append(results['rmse_test'])
            # Flatten and store the parameter arrays as strings to keep the CSV simple
            data['Ri_values'].append(', '.join(map(str, results['optimal_params']['Ri'])))
            data['Ci_values'].append(', '.join(map(str, results['optimal_params']['Ci'])))
            data['Ai_values'].append(', '.join(map(str, results['optimal_params']['Ai'])))
            data['Re_values'].append(', '.join(map(str, results['optimal_params']['Re'])))
            data['Ce_values'].append(', '.join(map(str, results['optimal_params']['Ce'])))
            data['Ae_values'].append(', '.join(map(str, results['optimal_params']['Ae'])))
            data['roxP_hvac'].append(', '.join(map(str, results['optimal_params']['roxP_hvac'])))

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # File name based on sensor count
        file_name = f"decent_R2C2_optimization_results_sensor_{sensor_count}.csv"

        # Save to CSV
        df.to_csv(file_name, index=False)
        print(f"Saved optimization results for sensor count {sensor_count} to '{file_name}'.")



