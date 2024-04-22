#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 15:54:29 2023

@author: ozanbaris
"""


import os
from scipy.linalg import expm, inv
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.optimize import NonlinearConstraint
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
class R2C2Cent:
    def __init__(self, sensor_count, R_values, Re_values, C_values, Ce_values, P_hvac, ro_values, A_values, A_e_values):
        """
        Initialize the RC model with the given parameters.
        
        Parameters:
        sensor_count (int): Number of sensors (internal zones) in the model.
        R_values (array): Thermal resistances between zones (K/W). Size is sensor_count x sensor_count.
        Re_values (array): External thermal resistances for each zone (K/W). Length is sensor_count.
        C_values (array): Thermal capacities (J/K) for internal zones. Length is sensor_count.
        Ce_values (array): Thermal capacities (J/K) for external zones. Length matches the number of external nodes.
        P_hvac (float): Heating/Cooling input of the HVAC system (W).
        ro_values (array): Heat distribution values across rooms. Length is sensor_count.
        A_values (array): Solar aperture coefficients for internal zones. Length is sensor_count.
        A_e_values (array): Solar aperture coefficients for external nodes. Length matches the number of external nodes.
        """
        self.sensor_count = sensor_count
        self.R_values = R_values
        self.Re_values = Re_values
        self.C_values = C_values
        self.Ce_values = Ce_values  # Added Ce_values for external thermal capacities
        self.P_hvac = P_hvac
        self.ro_values = ro_values
        self.A_values = A_values
        self.A_e_values = A_e_values
        
        self.N_states = sensor_count + len(Re_values)  # Total states including external nodes
        self.Ac = np.zeros((self.N_states, self.N_states))
        self.Bc = None
        
        self.update_matrices()
        
    def update_matrices(self):
        """Updates the model's matrices based on the current parameters."""

        # Update Ac matrix for internal zones
        for i in range(self.sensor_count):
            self.Ac[i, i] = -np.sum(1 / np.array(self.R_values[i])) / self.C_values[i] - 1 / self.Re_values[i] / self.C_values[i]
            for j in range(self.sensor_count):
                if i != j:
                    self.Ac[i, j] = 1 / self.R_values[i][j] / self.C_values[i]
    
        # Update Ac for interactions between internal and external zones
        for i in range(len(self.Re_values)):
            idx = self.sensor_count + i
            # Assuming Re_values[i] corresponds to the external resistance for zone i
            # and Ce_values for external capacities
            self.Ac[idx, idx] = - (1 / self.Re_values[i] + 1 / self.R_values[i][i]) / self.Ce_values[i]
            self.Ac[i, idx] = 1 / self.Re_values[i] / self.C_values[i]
            self.Ac[idx, i] = 1 / self.Re_values[i] / self.Ce_values[i]
        

        # Update Bc matrix
        self.Bc = np.zeros((self.N_states, 3))
        for i in range(self.sensor_count):
            self.Bc[i, 0] = -(self.P_hvac * self.ro_values[i] / self.C_values[i])
            self.Bc[i, 1] = self.A_values[i] / self.C_values[i]
            self.Bc[i, 2] = 1 / (self.R_values[i][i] * self.C_values[i])
        
        for i in range(len(self.A_e_values)):
            idx = self.sensor_count + i
            self.Bc[idx, 1] = self.A_e_values[i] / self.Ce_values[i]  # Corrected to use Ce_values
            self.Bc[idx, 2] = 1 / (self.Re_values[i] * self.Ce_values[i])  # Corrected to use Ce_values


        #print("Ac Matrix:")
        #print(self.Ac)
        #print("\nBc Matrix:")
        #print(self.Bc)
    def discretize(self, dt):
        """Discretizes the RC model given a timestep dt."""
        n = self.N_states
        # Discretizing the system dynamics
        F = expm(self.Ac * dt)
        # Integrating D into the discretization
        # Note: G calculation assumes B and D are concatenated for combined effect of control and disturbances
        try:
            # Attempt to invert the matrix
            G = np.dot(inv(self.Ac), np.dot(F - np.eye(n), self.Bc))
        except np.linalg.LinAlgError:
            # Handle the singular matrix case, perhaps by using a pseudoinverse or regularization
            G = np.dot(np.linalg.pinv(self.Ac), np.dot(F - np.eye(n), self.Bc))


        return F, G
    

    
    def predict(self, x, t, u, w):
        """
        Vectorized prediction when dt is constant for all time points, meaning F and G stay the same.
        
        Parameters:
        - x: Current state vectors at the start of each timestep (2D numpy array).
        - t: Time vector.
        - u: Control input vector over time (1D numpy array).
        - w: Weather vector over time (2D numpy array with weather variables as columns).
        
        Returns:
        - Predictions: Predicted state vectors for the next timestep (2D numpy array).
        """
       
        dt = 3600   # data is in hourly intervals.
        F, G = self.discretize(dt)  # Discretize the model once for constant dt
    
        # Prepare the inputs for vectorized operation
        u_expanded = u[:-1].reshape(-1, 1)  # Reshape u for proper broadcasting when combined
        combined_input = np.hstack((u_expanded, w[:-1]))  # Combine control and weather inputs, excluding the last element
    
        # Perform vectorized matrix multiplication

        predictions = np.dot(F, x.T) + np.dot(G, combined_input.T)
        # Transpose predictions to match expected shape (time steps as rows, sensor counts as columns)
        predictions = predictions.T
        return predictions[:, :self.sensor_count]

    def evaluate(self, parameters, df):
        """
        Evaluates the model by making predictions and comparing them to observed data.
        
        Parameters:
        - parameters: Dictionary containing model parameters.
        - df: DataFrame with observed data, including control inputs, weather data, and temperatures.
        
        Returns:
        - RMSE: Root Mean Square Error between the observed and predicted temperatures.
        """
        self.set_parameters(parameters)
        # Extract inputs from DataFrame
        u = df['duty_cycle'].values
        w = df[['GHI', 'Text']].values
        temp_columns = [f"T0{i}_TEMP" for i in range(1, self.sensor_count + 1)]
        outdoor_temp = df['Text'].values  # Outdoor temperatures
        y_obs = df[temp_columns].values[1:]  # Observed temperatures
        
        # Compute envelope temperatures as the average of sensor and outdoor temperatures
        sensor_temps = df[temp_columns].values[:-1]  # Sensor temperatures at initial states
        envelope_temps = (sensor_temps + outdoor_temp[:-1, None]) / 2  # Using broadcasting to match shapes
        
        x = np.hstack((sensor_temps,envelope_temps))  # Initial states updated to envelope temperatures
        t = np.arange(len(df))
        
        # Make predictions
        predictions = self.predict(x, t, u, w)
        
        # Calculate and return RMSE
        rmse = np.sqrt(mean_squared_error(y_obs, predictions))
        return rmse
    
    def set_parameters(self, parameters):
        """
        Updates the model parameters.
        
        Parameters:
        - parameters: Dictionary containing the model parameters.
        """
        self.R_values = parameters.get('R_values', self.R_values)
        self.C_values = parameters.get('C_values', self.C_values)
        self.A_values = parameters.get('A_values', self.A_values)
        self.Re_values = parameters.get('Re_values', self.R_values)
        self.Ce_values = parameters.get('Ce_values', self.C_values)
        self.A_e_values = parameters.get('A_e_values', self.A_values)
        self.P_hvac = parameters.get('P_hvac', self.P_hvac)
        self.ro_values = parameters.get('ro_values', self.ro_values)


        self.update_matrices()


def fitness_function(params, model, df):
    """
    Adjusted fitness function for the genetic algorithm with the updated parameter set.
    It evaluates the model with the given parameters against the observed data
    and returns the negative RMSE as the fitness value.
    """
    # Calculate the starting indices for each parameter in the params array
    R_values_end = model.sensor_count ** 2
    Re_values_end = R_values_end + model.sensor_count
    C_values_end = Re_values_end + model.sensor_count
    Ce_values_end = C_values_end + model.sensor_count
    A_values_end = Ce_values_end + model.sensor_count
    A_e_values_end = A_values_end + model.sensor_count
    ro_values_end = A_e_values_end + 1 + model.sensor_count
    
    #print(f"Expected params size: {model.sensor_count**2 + 6*model.sensor_count + 1}")
    #print(f"Actual params size: {len(params)}")
    #print(f"Index for P_hvac: {A_e_values_end}")

    # Adjust the model parameters based on the input
    param_dict = {
        'R_values': np.array(params[:R_values_end]).reshape(model.sensor_count, model.sensor_count),
        'Re_values': params[R_values_end:Re_values_end],
        'C_values': params[Re_values_end:C_values_end],
        'Ce_values': params[C_values_end:Ce_values_end],
        'A_values': params[Ce_values_end:A_values_end],
        'A_e_values': params[A_values_end:A_e_values_end],
        'P_hvac': params[A_e_values_end],
        'ro_values': params[A_e_values_end + 1:ro_values_end]
    }
    
    # Assume model.evaluate now accepts the new parameter dictionary and computes RMSE
    rmse = model.evaluate(param_dict, df)
    return rmse  # Assuming the goal is to minimize RMSE, hence return negative RMSE as fitness


def ro_values_constraint(params, model):
    """
    Constraint function to ensure the sum of 'ro_values' equals 1.
    
    Parameters:
    - params: The full parameter array from the optimization algorithm.
    - model: The model instance, used to determine the sensor count.
    
    Returns:
    - The difference between the sum of 'ro_values' and 1, which should be 0 when the constraint is met.
    """
    # Calculate the ending indices based on the model's sensor count
    R_values_end = model.sensor_count ** 2
    Re_values_end = R_values_end + model.sensor_count
    C_values_end = Re_values_end + model.sensor_count
    Ce_values_end = C_values_end + model.sensor_count
    A_values_end = Ce_values_end + model.sensor_count
    A_e_values_end = A_values_end + model.sensor_count
    ro_values_start = A_e_values_end + 1  # Start index for ro_values

    # Calculate the sum of 'ro_values'
    ro_values_sum = np.sum(params[ro_values_start:ro_values_start + model.sensor_count])

    # Constraint function should return 0 when the sum of 'ro_values' is exactly 1
    return ro_values_sum - 1




def optimize_parameters(model, df, bounds):
    """
    Optimizes the parameters of the RCModel using differential evolution,
    followed by local refinement with constraints using SLSQP.
    """
    def fitness_wrapper(params):
        return fitness_function(params, model, df)

    # Create a NonlinearConstraint object for 'ro_values' to sum to 1
    ro_constraint = NonlinearConstraint(ro_values_constraint, lb=1, ub=1)
    # Global optimization with differential evolution
    result_ga = differential_evolution(fitness_wrapper, bounds, strategy='best1bin', maxiter=40, popsize=15, tol=0.01, mutation=(0.2, 1.8), recombination=0.7, disp=True)

    # Adjust the constraint to pass 'model' to 'ro_values_constraint'
    ro_constraint_dict = {
        'type': 'eq',  # Specifies that this is an equality constraint
        'fun': lambda params: ro_values_constraint(params, model)
    }
    # Local refinement with SLSQP, including the constraint
    initial_guess = result_ga.x
    result_slsqp = minimize(fitness_wrapper, initial_guess, method='SLSQP', bounds=bounds, constraints=[ro_constraint_dict], options={'maxiter': 100, 'disp': True})
    print(f"Optimized RMSE: {result_slsqp.fun}")

    return result_slsqp.x, result_slsqp.fun


def set_r2c2_model_from_distribution(sensor_count, bounds):
    """
    Creates an R2C2Cent instance with parameters selected from uniform distributions within specified bounds.
    The bounds list now contains tuples for the bounds of R_values, Re_values, C_values, Ce_values, A_values, A_e_values, and P_hvac,
    in that order. The length of bounds should be 6, assuming A_e_values share bounds with A_values.
    
    Parameters:
    - sensor_count (int): The number of internal sensors (and thus the length of R_values, C_values, A_values, ro_values).
    - bounds (list of tuple): Each tuple contains the (min, max) bounds for the parameters, including external nodes.
    
    Returns:
    - R2C2Cent instance with randomly selected parameters.
    """
    R_values = np.random.uniform(bounds[0][0], bounds[0][1], size=(sensor_count, sensor_count)).tolist()
    Re_values = [np.random.uniform(bounds[1][0], bounds[1][1]) for _ in range(sensor_count)]
    C_values = [np.random.uniform(bounds[2][0], bounds[2][1]) for _ in range(sensor_count)] + \
               [np.random.uniform(bounds[3][0], bounds[3][1]) for _ in range(sensor_count)]  # Including external nodes
    A_values = [np.random.uniform(bounds[4][0], bounds[4][1]) for _ in range(sensor_count)]
    A_e_values = [np.random.uniform(bounds[4][0], bounds[4][1]) for _ in range(sensor_count)]  # Assuming same bounds for A and A_e
    P_hvac = np.random.uniform(bounds[5][0], bounds[5][1])  # Assuming a single value for simplicity
    ro_values = [1.0 / sensor_count for _ in range(sensor_count)]  # Uniform distribution of heat
    
    r2c2_model = R2C2Cent(sensor_count=sensor_count, 
                          R_values=R_values, 
                          Re_values=Re_values,
                          C_values=C_values[:sensor_count],  # Only internal C_values
                          Ce_values=C_values[sensor_count:],  # Only external Ce_values
                          P_hvac=P_hvac, 
                          ro_values=ro_values,
                          A_values=A_values, 
                          A_e_values=A_e_values)
    
    return r2c2_model

def optimize_for_sensor_count(processed_houses, add_sensor_count, optimization_results, bounds):
    """
    Optimizes each house for a given sensor count and updates the optimization results dictionary
    to be nested by sensor_count first, then house_id, for the R2C2Cent with external nodes.
    """
    # Ensure a nested dictionary exists for this sensor count
    if add_sensor_count not in optimization_results:
        optimization_results[add_sensor_count] = {}
    
    houses = processed_houses[add_sensor_count]
    for house_id, df in houses.items():
        sensor_count = add_sensor_count + 1 
        print('Sensor count:', sensor_count, '| House ID:', house_id)
        
        # Calculate the test dataset size
        test_size = 0.125
        num_test_samples = int(len(df) * test_size)
        split_index = len(df) - num_test_samples
        
        # Split the DataFrame without shuffling
        train_df = df.iloc[:split_index]
        test_df = df.iloc[split_index:]
        
        # Initialize the R2C2 model with random parameters within the bounds
        model = set_r2c2_model_from_distribution(sensor_count, bounds)
        
        # Optimize parameters for the model based on the training dataset
        optimal_params_array, optimal_rmse_train = optimize_parameters(model, train_df, bounds)
        
        # Assuming the returned optimal_params_array has the correct order and length
        # Convert the array to a dictionary for optimized parameters
        R_values_end = sensor_count ** 2
        Re_values_end = R_values_end + sensor_count
        C_values_end = Re_values_end + sensor_count
        Ce_values_end = C_values_end + sensor_count
        A_values_end = Ce_values_end + sensor_count
        A_e_values_end = A_values_end + sensor_count
        ro_values_start = A_e_values_end + 1
        
        optimal_params_dict = {
            'R_values': np.array(optimal_params_array[:R_values_end]).reshape(sensor_count, sensor_count),
            'Re_values': optimal_params_array[R_values_end:Re_values_end],
            'C_values': optimal_params_array[Re_values_end:C_values_end],
            'Ce_values': optimal_params_array[C_values_end:Ce_values_end],
            'A_values': optimal_params_array[Ce_values_end:A_values_end],
            'A_e_values': optimal_params_array[A_values_end:A_e_values_end],
            'P_hvac': optimal_params_array[A_e_values_end],
            'ro_values': optimal_params_array[ro_values_start:ro_values_start + sensor_count]
        }
        
        print(optimal_params_dict['ro_values'])
        # Set model parameters based on optimization
        model.set_parameters(optimal_params_dict)
        
        # Evaluate the model with the test dataset
        rmse_test = model.evaluate(optimal_params_dict, test_df)
        
        # Update the results in the nested dictionary
        optimization_results[add_sensor_count][house_id] = {
            'optimal_params': optimal_params_dict,
            'rmse_train': optimal_rmse_train,
            'rmse_test': rmse_test
        }


#%%


def fit_models_R2C2Cent(processed_houses_reduced):
    # Initialize the dictionary to store optimization results
    optimization_results_R2C2_cent = {}

    # Start timing
    start_time = time.time()

    for i in range(5):
        add_sensor_count = 1 + i
        sensor_count = add_sensor_count + 1
        
        bounds = (
            *[(0.000001, 0.05)] * (sensor_count**2),  # R_values bounds for internal zones
            *[(0.00001, 0.1)] * sensor_count,  # Re_values bounds for external resistances
            *[(10000, 10000000)] * sensor_count,  # C_values bounds for internal nodes
            *[(1000, 10000000)] * sensor_count,  # Ce_values bounds for external nodes
            *[(0.00001, 1.5)] * sensor_count,  # A_values bounds for internal nodes
            *[(0.00001, 1.5)] * sensor_count,  # A_e_values bounds for external nodes
            (2000, 15000),  # P_hvac bound
            *[(0.01, 0.99)] * sensor_count,  # ro_values bounds
        )

        optimize_for_sensor_count(processed_houses_reduced, add_sensor_count, optimization_results_R2C2_cent, bounds)

    # End timing
    end_time = time.time()

    # Calculate and print the execution time
    execution_time_minutes = (end_time - start_time) / 60
    print(f"The loop took {execution_time_minutes:.2f} minutes to run.")

    return optimization_results_R2C2_cent


def save_results_R2C2Cent(optimization_results):
    """
    Saves optimization results to CSV, with a separate file for each sensor count.

    Parameters:
    - optimization_results: A dictionary with nested structure [sensor_count][house_id] containing optimization outcomes.
    """
    for sensor_count, houses in optimization_results.items():
        # Initialize lists to store the data
        data = {
            'House ID': [],
            'RMSE Train': [],
            'RMSE Test': [],
            # Parameters
            'R_values': [],
            'Re_values': [],
            'C_values': [],
            'Ce_values': [],
            'A_values': [],
            'Ae_values': [],
            'P_hvac': [],
            'ro_values': [],
        }


        # Populate the data structure
        for house_id, results in houses.items():
            data['House ID'].append(house_id)
            data['RMSE Train'].append(results['rmse_train'])
            data['RMSE Test'].append(results['rmse_test'])
            # Flatten and store the arrays as strings to keep the CSV simple
            data['R_values'].append(', '.join(map(str, results['optimal_params']['R_values'])))
            data['Re_values'].append(', '.join(map(str, results['optimal_params']['Re_values'])))
            data['C_values'].append(', '.join(map(str, results['optimal_params']['C_values'])))
            data['Ce_values'].append(', '.join(map(str, results['optimal_params']['Ce_values'])))
            data['A_values'].append(', '.join(map(str, results['optimal_params']['A_values'])))
            data['Ae_values'].append(', '.join(map(str, results['optimal_params']['A_e_values'])))
            data['P_hvac'].append(results['optimal_params']['P_hvac'])
            data['ro_values'].append(', '.join(map(str, results['optimal_params']['ro_values'])))

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # File name based on sensor count
        file_name = f"cent_R2C2_optimization_results_sensor_{sensor_count}.csv"

        # Save to CSV
        df.to_csv(file_name, index=False)
        print(f"Saved optimization results for sensor count {sensor_count} to '{file_name}'.")
