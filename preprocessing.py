#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 15:54:29 2023

@author: ozanbaris
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


def read_csvs_to_dfs(main_output_directory):
    all_houses_dict = {}
    
    # Iterate over each subdirectory within the main directory
    for subdirectory in os.listdir(main_output_directory):
        sub_output_directory = os.path.join(main_output_directory, subdirectory)
        
        # Skip if it's not a directory
        if not os.path.isdir(sub_output_directory):
            continue
        
        # Extract house group from the subdirectory name and convert to integer
        house_group = int(subdirectory.split("_")[-1])
        
        # Initialize the dictionary for this house group if it doesn't exist
        if house_group not in all_houses_dict:
            all_houses_dict[house_group] = {}
        
        # Iterate over each CSV file within the subdirectory
        for filename in os.listdir(sub_output_directory):
            if filename.endswith(".csv"):
                # Construct the full file path
                file_path = os.path.join(sub_output_directory, filename)
                
                # Extract house_id from the filename
                house_id = filename.split("_")[-1].replace(".csv", "")
                
                # Read the CSV file into a DataFrame
                df = pd.read_csv(file_path)
                
                # Store the DataFrame in the dictionary under the correct house group
                all_houses_dict[house_group][house_id] = df
                
    return all_houses_dict

def process_house_data(df):
    """
    Processes a single house data DataFrame with the specified operations.
    """
    # (1) Normalize CoolingRunTime to hours
    df['duty_cycle'] = df['CoolingRunTime'] / 3600

    # (2) Rename the outdoor temperature column to Text
    df.rename(columns={'Outdoor_Temperature': 'Text'}, inplace=True)

    # (3) Rename the sensor columns
    sensor_rename_map = {
        'Thermostat_Temperature': 'T01_TEMP',
        'RemoteSensor1_Temperature': 'T02_TEMP',
        'RemoteSensor2_Temperature': 'T03_TEMP',
        'RemoteSensor3_Temperature': 'T04_TEMP',
        'RemoteSensor4_Temperature': 'T05_TEMP',
        'RemoteSensor5_Temperature': 'T06_TEMP',
    }
    df.rename(columns=sensor_rename_map, inplace=True)

    # (4) Convert temperature columns from Fahrenheit to Kelvin
    temp_columns = [f"T0{i}_TEMP" for i in range(1, 7)] + ['Text']
    for col in temp_columns:
        df[col] = (df[col] - 32) * 5/9 + 273.15

    # (5) Keep only the necessary columns
    columns_to_keep = ['time', 'GHI',  'duty_cycle'] + temp_columns
    df = df[columns_to_keep]

    # (6) Forward fill to handle missing values
    df.fillna(method='ffill', inplace=True)

    return df

def print_optimization_statistics(optimization_results):
    """
    Prints the statistics of the optimization results (training and testing RMSE)
    for each sensor count.

    Parameters:
    - optimization_results: Nested dictionary with optimization results organized by sensor_count.
    """
    for sensor_count, houses in optimization_results.items():
        # Extracting RMSE values for training and testing
        rmse_train = [details['rmse_train'] for details in houses.values()]
        rmse_test = [details['rmse_test'] for details in houses.values()]
        
        # Using NumPy to calculate statistics
        print(f"Sensor Count: {sensor_count}")
        print("Training RMSE Statistics:")
        print(f"  Mean: {np.mean(rmse_train):.2f}")
        print(f"  Median: {np.median(rmse_train):.2f}")
        print(f"  Max: {np.max(rmse_train):.2f}")
        print(f"  Min: {np.min(rmse_train):.2f}")
        print(f"  Standard Deviation: {np.std(rmse_train):.2f}\n")
        
        print("Testing RMSE Statistics:")
        print(f"  Mean: {np.mean(rmse_test):.2f}")
        print(f"  Median: {np.median(rmse_test):.2f}")
        print(f"  Max: {np.max(rmse_test):.2f}")
        print(f"  Min: {np.min(rmse_test):.2f}")
        print(f"  Standard Deviation: {np.std(rmse_test):.2f}\n")



def plot_error_distribution(optimization_results):
    """
    Plots the boxplot of the distribution of test RMSE errors for each sensor count.
    
    Parameters:
    - optimization_results: A dictionary with nested structure [sensor_count][house_id] containing optimization outcomes.
    """
    # Prepare data for plotting
    data = {'Sensor Count': [], 'Test RMSE': []}
    for sensor_count, houses in optimization_results.items():
        for house_id, results in houses.items():
            data['Sensor Count'].append(sensor_count)
            data['Test RMSE'].append(results['rmse_test'])
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.ylim(0,0.6)
    plt.title('Distribution of Test RMSE Errors by Sensor Count')
    sns.boxplot(x='Sensor Count', y='Test RMSE', data=df)
    plt.xlabel('Sensor Count')
    plt.ylabel('Test RMSE')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

