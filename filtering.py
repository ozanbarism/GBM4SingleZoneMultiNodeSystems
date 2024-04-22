#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 16:58:14 2024

@author: ozanbaris
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def collect_and_save_at_bounds(optimization_results):
    # Define the bounds matching those used in your model's optimization function
    bounds = [
        (0.00001, 0.05),  # Ri bounds
        (0.00001, 0.1),   # Re bounds
        (10000, 10000000), # Ci bounds
        (10000, 10000000), # Ce bounds
        (0.00001, 1.5),   # Ai bounds
        (0.00001, 1.5),   # Ae bounds
        (200, 10000)      # roxP_hvac bounds
    ]
    # Define parameter names
    param_names = ['Ri', 'Re', 'Ci', 'Ce', 'Ai', 'Ae', 'roxP_hvac']

    # Create a dictionary of dataframes to collect rows where bounds are hit
    dataframes = {param: [] for param in param_names}

    # Iterate over all results
    for add_sensor_count, houses in optimization_results.items():
        for house_id, results in houses.items():
            optimal_params = results['optimal_params']
            rmse_train = results['rmse_train']
            rmse_test = results['rmse_test']

            # Only process data for houses where RMSE test is less than 0.5
            if rmse_test < 0.5:
                # Iterate over all sensors in the house data
                for sensor_index, params in enumerate(zip(*[optimal_params[p] for p in param_names]), start=1):
                    # params will be a tuple containing one value for each parameter from the current sensor
                    for idx, value in enumerate(params):
                        param = param_names[idx]
                        lower_bound, upper_bound = bounds[idx]
                        if value == lower_bound or value == upper_bound:
                            # Collect all other parameter values from the same sensor index
                            data_row = {
                                'sensor_count': add_sensor_count + 1,
                                'house_id': house_id,
                                'sensor_index': sensor_index,
                                'rmse_train': rmse_train,
                                'rmse_test': rmse_test
                            }
                            # Include values for all parameters
                            for j, p in enumerate(param_names):
                                data_row[p] = params[j]  # params[j] is the value of parameter p at the current sensor index

                            dataframes[param].append(data_row)

    # Convert lists to DataFrames and save to CSV
    for param, data_list in dataframes.items():
        if data_list:
            df = pd.DataFrame(data_list)
            df.to_csv(f'{param}_at_bounds.csv', index=False)
            print(f"Saved data for parameter {param} at bounds to '{param}_at_bounds.csv'.")





#%%


def enrich_data(results_file, processed_houses_reduced, ri_upper_bound, re_upper_bound):
    # Load results with house IDs at bounds
    results = pd.read_csv(results_file)
    
    # Prepare columns for state, mean outdoor temperature, mean temperature difference, and total cooling runtime
    results['State'] = ''
    results['Mean_Outdoor_Temperature'] = 0.0
    results['Mean_Temperature_Difference'] = 0.0
    results['Total_CoolingRunTime'] = 0
    
    # Mapping sensor index to temperature column names
    sensor_columns = {
        1: 'Thermostat_Temperature',
        2: 'RemoteSensor1_Temperature',
        3: 'RemoteSensor2_Temperature',
        4: 'RemoteSensor3_Temperature',
        5: 'RemoteSensor4_Temperature',
        6: 'RemoteSensor5_Temperature'
    }

    # Initialize a list to collect indices of rows meeting the condition
    upper_bound_indices = []

    # Iterate through the results to enrich with state and temperature differences and cooling runtime
    for idx, row in results.iterrows():
        # Check if both Ri and Re are at their upper bounds
        if row['Ri'] == ri_upper_bound:
            upper_bound_indices.append(idx)
            house_id = row['house_id']
            found = False

            # Search for the house in processed_houses_reduced
            for add_sensor_count, houses in processed_houses_reduced.items():
                if house_id in houses:
                    df = houses[house_id]
                    sensor_index = int(row['sensor_index'])
                    sensor_col = sensor_columns[sensor_index]  # Get the sensor column name based on index

                    # Calculate the mean temperature difference and total cooling runtime
                    if sensor_col in df.columns and 'Outdoor_Temperature' in df.columns and 'CoolingRunTime' in df.columns:
                        temperature_difference = df['Outdoor_Temperature'] - df[sensor_col]
                        mean_temperature_difference = temperature_difference.mean()
                        total_cooling_runtime = df['CoolingRunTime'].sum()  # Sum the cooling runtime
                        total_ghi=df['GHI'].sum() 
                        # Assign the state, temperatures, and cooling runtime to the results DataFrame
                        results.at[idx, 'State'] = df['State'].iloc[0]
                        results.at[idx, 'Mean_Outdoor_Temperature'] = df['Outdoor_Temperature'].mean()
                        results.at[idx, 'Mean_Temperature_Difference'] = mean_temperature_difference
                        results.at[idx, 'Total_CoolingRunTime'] = total_cooling_runtime
                        results.at[idx, 'GHI_total'] = total_ghi
                        found = True
                        break

            if not found:
                print(f"House ID {house_id} not found in processed data or missing necessary columns.")

    # Filter the DataFrame to only include rows where Ri and Re are at upper bounds
    filtered_results = results.loc[upper_bound_indices]

    # Save the filtered results back to CSV
    filtered_results.to_csv('Ri_Re_at_upper_bounds_enriched.csv', index=False)
    print(f"Updated results saved to 'Ri_Re_at_upper_bounds_enriched.csv'.")
    print(f"Number of unique houses with Ri and Re at upper bounds: {filtered_results['house_id'].nunique()}.")



