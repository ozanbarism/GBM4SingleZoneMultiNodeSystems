#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 10:14:10 2024

@author: ozanbaris
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

def aggregate_rmse_data(sensor_counts):
    # Define the base file names for the four file formats
    file_bases = [
        "RESULTS/decent_R1C1_optimization_results_sensor_{sensor_count}",
        "RESULTS/decent_R2C2_optimization_results_sensor_{sensor_count}",
        "RESULTS/cent_R1C1_optimization_results_sensor_{sensor_count}",
        "RESULTS/cent_R2C2_optimization_results_sensor_{sensor_count}"
    ]
    
    # List to collect data rows
    data_rows = []
    
    # Iterate over each sensor count
    for sensor_count in sensor_counts:
        # Iterate over each file base format
        for file_base in file_bases:
            # Format the file name with the current sensor count
            file_name = f"{file_base}.csv".format(sensor_count=sensor_count)
            # Check if the file exists
            if os.path.exists(file_name):
                # Read the CSV file
                df = pd.read_csv(file_name)
                # Extract the optimization type and configuration from the file base
                type_, configuration = file_base.split("/")[1].split("_")[0], "_".join(file_base.split("_")[1:3])
                # Append the necessary data to the data_rows list
                for _, row in df.iterrows():
                    data_rows.append({
                        "Sensor Count": sensor_count,
                        "Type": type_,
                        "Configuration": configuration,
                        "RMSE Train": row["RMSE Train"],
                        "RMSE Test": row["RMSE Test"]
                    })
            else:
                print(f"File '{file_name}' not found.")
    
    # Create DataFrame from collected rows
    aggregated_data = pd.DataFrame(data_rows, columns=["Sensor Count", "Type", "Configuration", "RMSE Train", "RMSE Test"])
    
    return aggregated_data


# Example usage
sensor_counts = [1, 2, 3, 4, 5]  # Update this list based on the actual sensor counts you have
aggregated_data = aggregate_rmse_data(sensor_counts)
#%%

def plot_rmse_distribution(aggregated_data):
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))

    # Correct the 'Category' column creation to ensure proper label formatting
    aggregated_data['Category'] = aggregated_data.apply(lambda row: f"{row['Type']} {row['Configuration'].split('_')[0]}", axis=1)
    
    # Adjusted palette with unique colors for each category, changing green to red
    palette = {
        'cent R1C1': 'tab:blue',
        'cent R2C2': 'lightblue',
        'decent R1C1': 'tab:red',  # Changed from green to red
        'decent R2C2': 'lightcoral',  # Changed from lightgreen to lightcoral
    }

    # Plot the boxplot
    sns.boxplot(x='Sensor Count', y='RMSE Test', hue='Category', data=aggregated_data, palette=palette, showfliers=False)

    # Adjust legend labels
    plt.legend(bbox_to_anchor=(0.17, 1.01), loc=2, ncols=4, fontsize=14)

    # Adjust the fontsize for the xticks and yticks and their titles
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('Sensor Count', fontsize=18)
    plt.ylabel('RMSE Test (K)', fontsize=18)
    plt.savefig('errors.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

# Assuming 'aggregated_data' is your DataFrame
plot_rmse_distribution(aggregated_data)


