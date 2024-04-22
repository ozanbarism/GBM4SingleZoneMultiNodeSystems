#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 15:54:29 2023

@author: ozanbaris
"""


#%% READ THE DATA
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
import numpy as np
from scipy.linalg import expm, inv
from scipy.optimize import differential_evolution, minimize
from sklearn.metrics import mean_squared_error
import time

class R1C1Decent:
    def __init__(self, R_value, C_value, A_value, roxP_hvac):
        """
        Initialize the 1R1C model with the given parameters.
        
        Parameters:
        R_value (float): Thermal resistance (K/W).
        C_value (float): Thermal capacity (J/K).
        A_value (float): Solar aperture coefficient.
        P_hvac (float): Power input of the HVAC system.
        """
        self.R = R_value
        self.C = C_value
        self.A = A_value
        self.roxP_hvac = roxP_hvac
        
    def predict(self, T, T_ext, u, ghi,  dt):
        """
        Update the indoor temperatures for the next time step using the Euler method in a vectorized form.
        
        Parameters:
        T (np.array): Current indoor temperatures (K) for each time step or sensor.
        T_ext (np.array): External temperatures (K) corresponding to each time step or sensor.
        roxP_hvac (float): ratio x Power input from the HVAC system (W).
        u (np.array): Duration HVAC was on (duty cycle) for each time step or sensor.
        ghi (np.array): Solar radiation (W/m2) for each time step or sensor.
        R (float): Thermal resistance (K/W).
        C (float): Thermal capacity (Wh/K).
        A (float): Solar aperture coefficient assumed to include area (m2).
        delta_t (float): Time step (h).
        
        Returns:
        np.array: Temperatures at the next time step (F) for each time step or sensor.
        """
        # Compute the change in temperature due to external temperature, HVAC power, and solar gain
        R=self.R
        C=self.C
        A=self.A
        roxP_hvac=self.roxP_hvac
        
        delta_T_ext = (T_ext - T) / (R * C)
        delta_T_hvac = -(roxP_hvac * u) / C
        delta_T_solar = (A * ghi) / C
        
        # Total change in temperature for each time step or sensor
        delta_T_total = delta_T_ext + delta_T_hvac + delta_T_solar
        
        # Update the temperature for the next time step
        T_next = T + dt * delta_T_total
        #print(T_next)
        return T_next


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
        u = df['duty_cycle'].values[:-1]  # HVAC duty cycle
        ghi = df['GHI'].values[:-1]  # Solar radiation
        
        # Predict the next temperatures
        predictions = self.predict(T, T_ext, u, ghi, dt)
        
        # Observed temperatures for comparison (excluding the first value used as initial condition)
        y_obs = df[sensor].values[1:]
        
        # Calculate RMSE between observed temperatures and predictions
        rmse = np.sqrt(mean_squared_error(y_obs, predictions))
        
        return rmse
    def set_parameters(self, params):
        """
        Sets the parameters of the model based on the input dictionary.
        """
        self.R = params['R']
        self.C = params['C']
        self.A = params['A']
        self.roxP_hvac = params['roxP_hvac']

def fitness_function(params, model, df, sensor):
    """
    Fitness function for the optimization algorithm.
    It evaluates the model with the given parameters against the observed data
    and returns the negative RMSE as the fitness value to be minimized.
    """
    param_dict = {
        'R': params[0],
        'C': params[1],
        'A': params[2],
        'roxP_hvac': params[3],
    }
    
    # Set model parameters
    model.set_parameters(param_dict)
    
    # Assuming 'sensor' column in df holds the sensor temperature data
    # and df is properly structured with 'duty_cycle', 'GHI', and 'Text' columns
    rmse = model.evaluate(param_dict, sensor, df)
    
    return rmse  # Minimize the negative RMSE

def optimize_parameters(model, df, sensor, bounds):
    """
    Optimizes the parameters of the R1C1Decent.
    """

    def fitness_wrapper(params):
        return fitness_function(params, model, df, sensor)

    result_ga = differential_evolution(fitness_wrapper, bounds, strategy='best1bin', 
                                       maxiter=100, popsize=15, tol=0.001, 
                                       mutation=(0.2, 1.8), recombination=0.7, disp=True)
    
    
    #print(f"Optimized Parameters after GA: {result_ga.x}")
    print(f"Optimized  RMSE after GA: {result_ga.fun}")

    # Prepare for SLSQP refinement
    initial_guess = result_ga.x

    # Sequential Least Squares Programming (SLSQP) optimization for local refinement
    result_slsqp = minimize(fitness_wrapper, initial_guess, method='SLSQP', bounds=bounds, 
                            #constraints=[ro_constraint],  # Uncomment if you have defined constraints
                            options={'maxiter': 100, 'disp': True})
    
    #print(f"Refined Optimized Parameters: {result_slsqp.x}")
    print(f"Refined Optimized  RMSE: {result_slsqp.fun}")

    optimized_dict = {
    'R': result_slsqp.x[0],
    'C': result_slsqp.x[1],
    'A': result_slsqp.x[2],
    'roxP_hvac': result_slsqp.x[3],
    }

    return optimized_dict, result_slsqp.fun


def set_rc_model_from_distribution(sensor_count, bounds):

    # Selecting parameters from the uniform distribution within the specified bounds
    R_value = np.random.uniform(bounds[0][0], bounds[0][1])
    C_value = np.random.uniform(bounds[1][0], bounds[1][1])
    A_value = np.random.uniform(bounds[2][0], bounds[2][1])
    roxP_hvac = np.random.uniform(bounds[3][0], bounds[3][1])

    # Creating an RCModel instance with the randomly selected parameters
    rc_model = R1C1Decent(R_value=R_value, 
                       C_value=C_value, 
                       A_value=A_value, 
                       roxP_hvac=roxP_hvac)
    
    return rc_model

def fit_models_R1C1Decent(processed_houses_reduced):
    optimization_results_decent = {}
    # Start timing
    start_time = time.time()
    for add_sensor_count, houses in processed_houses_reduced.items():
        optimization_results_decent[add_sensor_count] = {}
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

            aggregated_params = {'R': [], 'C': [], 'A': [], 'roxP_hvac': []}
            
            rmse_test_collect = []
            rmse_train_collect = []
            for sensor_index in range(1, sensor_count + 1):
                sensor_column = f"T0{sensor_index}_TEMP"
                bounds = [
                    (0.00001, 0.3),  # R bounds
                    (10000, 10000000),  # C bounds
                    (0.00001, 1.5),  # A bound
                    (200, 10000)  # roxP_hvac bound
                ]

                rc_model = set_rc_model_from_distribution(sensor_count, bounds)
                
                optimal_params, rmse_train = optimize_parameters(rc_model, train_df, sensor_column, bounds)
                
                if optimal_params is not None:
                    aggregated_params['R'].append(optimal_params['R'])
                    aggregated_params['C'].append(optimal_params['C'])
                    aggregated_params['A'].append(optimal_params['A'])
                    aggregated_params['roxP_hvac'].append(optimal_params['roxP_hvac'])

                    # Set model parameters based on optimization
                    rc_model.set_parameters(optimal_params)

                    # Evaluate the model with the test dataset
                    rmse_test = rc_model.evaluate(optimal_params, sensor_column, test_df)

                    rmse_test_collect.append(rmse_test)
                    rmse_train_collect.append(rmse_train)
                else:
                    # If optimization failed for a sensor, handle it accordingly
                    print(f"Optimization failed for sensor {sensor_column} in house {house_id}. Skipping this sensor.")
                    continue

            # Save the results
            optimization_results_decent[add_sensor_count][house_id] = {
                'optimal_params': aggregated_params,
                'rmse_train': np.mean(rmse_train_collect),
                'rmse_test': np.mean(rmse_test_collect)
            }
        # End timing
    end_time = time.time()

    # Calculate and print the execution time
    execution_time_minutes = (end_time - start_time) / 60
    print(f"The loop took {execution_time_minutes:.2f} minutes to run.")

    return optimization_results_decent


#%%


def plot_parameter_distributions_R1C1Decent(optimization_results):
    parameters = ['R', 'C', 'A', 'roxP_hvac']
    parameter_titles = ['R Values Distribution', 'C Values Distribution', 'A Values Distribution', 'roxP_hvac Distribution']
    colors = ['blue', 'green', 'red', 'purple', 'orange']  # Different color for each sensor count

    # Setting up the plot
    fig, axs = plt.subplots(4, 1, figsize=(10, 20))
    sensor_counts = sorted(optimization_results.keys())

    for param_index, param in enumerate(parameters):
        for sensor_idx, sensor_count in enumerate(sensor_counts):
            # Aggregate parameter values from all houses for this sensor count
            all_values = []
            houses = optimization_results[sensor_count]
            for house_id, results in houses.items():
                all_values.extend(results['optimal_params'][param])
            
            # Plot histogram for the parameter values of this sensor count
            sns.histplot(all_values, kde=False, ax=axs[param_index], stat="density", label=f"Sensor Count {sensor_count} - Histogram", color=colors[sensor_idx % len(colors)], bins=100, alpha=0.5, edgecolor='black')
            
            # Overlay a KDE plot (normal distribution curve) for this sensor count
            sns.kdeplot(all_values, ax=axs[param_index], color=colors[sensor_idx % len(colors)], label=f"Sensor Count {sensor_count} - KDE")

        axs[param_index].set_title(parameter_titles[param_index])
        axs[param_index].set_ylabel('Density')
        axs[param_index].legend()
    
    plt.tight_layout()
    plt.show()

#%%

def save_results_R1C1Decent(optimization_results_decent):
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
            'R_values': [],
            'C_values': [],
            'A_values': [],
            'roxP_hvac': []
        }

        # Populate the data structure
        for house_id, results in houses.items():
            data['House ID'].append(house_id)
            data['RMSE Train'].append(results['rmse_train'])
            data['RMSE Test'].append(results['rmse_test'])
            # Flatten and store the parameter arrays as strings to keep the CSV simple
            data['R_values'].append(', '.join(map(str, results['optimal_params']['R'])))
            data['C_values'].append(', '.join(map(str, results['optimal_params']['C'])))
            data['A_values'].append(', '.join(map(str, results['optimal_params']['A'])))
            data['roxP_hvac'].append(', '.join(map(str, results['optimal_params']['roxP_hvac'])))

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # File name based on sensor count
        file_name = f"decent_R1C1_optimization_results_sensor_{sensor_count}.csv"

        # Save to CSV
        df.to_csv(file_name, index=False)
        print(f"Saved optimization results for sensor count {sensor_count} to '{file_name}'.")




