#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 15:54:29 2023

@author: ozanbaris
"""


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import expm, inv
from scipy.stats import norm


#%%

def filter_houses_and_save_remaining(file_path):
    # Load the CSV file
    results = pd.read_csv(file_path)
    
    # Define the condition to filter out houses
    conditions = (
        (results['Re'] == 0.1) 
    )
    
    # Identify houses to be filtered
    filtered_houses = results[conditions]['house_id'].tolist()

    # Create a DataFrame of remaining houses
    remaining_results = results[~conditions]

    # Save the remaining results to a new CSV file
    remaining_results.to_csv('remaining_houses.csv', index=False)
    print("Remaining houses saved to 'remaining_houses.csv'.")
    print("Houses to be filtered based on temperature difference criteria:", len(filtered_houses))

    return filtered_houses


filtered_houses = filter_houses_and_save_remaining('Ri_Re_at_upper_bounds_enriched.csv')


#%%

def read_optimization_results_from_csv(sensor_counts, filtered_houses):
    """
    Reads optimization results from CSV files into a dictionary, filtering out houses based on two conditions:
    1. An RMSE test value larger than 0.5.
    2. House IDs listed in filtered_houses due to temperature difference criteria.

    Parameters:
    - sensor_counts: A list of sensor counts to identify which CSV files to read.
    - filtered_houses: List of house IDs to filter out based on external criteria.

    Returns:
    A dictionary with the nested structure [sensor_count][house_id] containing optimization outcomes,
    excluding those with RMSE Test > 0.5 and houses listed in filtered_houses.
    """
    optimization_results = {}
    for sensor_count in sensor_counts:
        file_name = f"RESULTS/decent_R2C2_optimization_results_sensor_{sensor_count}.csv"
        if os.path.exists(file_name):
            df = pd.read_csv(file_name)
            
            initial_house_count = len(df)
            # Apply both RMSE test filter and house ID filter
            df_filtered = df[(df['RMSE Test'] <= 0.5) & (~df['House ID'].isin(filtered_houses))]
            filtered_house_count = len(df_filtered)
            
            print(f"Sensor Group {sensor_count}: {initial_house_count} houses before filtering, {filtered_house_count} houses after filtering.")

            houses = {}
            for _, row in df_filtered.iterrows():
                house_id = row['House ID']
                results = {
                    'rmse_train': row['RMSE Train'],
                    'rmse_test': row['RMSE Test'],
                    'optimal_params': {
                        'Ri': list(map(float, row['Ri_values'].split(', '))),
                        'Re': list(map(float, row['Re_values'].split(', '))),
                        'Ci': list(map(float, row['Ci_values'].split(', '))),
                        'Ce': list(map(float, row['Ce_values'].split(', '))),
                        'Ai': list(map(float, row['Ai_values'].split(', '))),
                        'Ae': list(map(float, row['Ae_values'].split(', '))),
                        'roxP_hvac': list(map(float, row['roxP_hvac'].split(', ')))
                    }
                }
                houses[house_id] = results
            
            optimization_results[sensor_count] = houses
        else:
            print(f"File '{file_name}' not found.")
    
    return optimization_results


sensor_counts = [1, 2, 3, 4, 5] 
optimization_results_R2C2_decent = read_optimization_results_from_csv(sensor_counts, filtered_houses)

#%%

def plot_parameter_distributions(optimization_results):
    parameters = ['Ri', 'Re', 'Ci', 'Ai', 'roxP_hvac']
    parameter_x_labels = ['$R_{i}$ (K/W)', '$R_{e}$ (K/W)', 
                          '$C_{i}$ (J/K)', '$\\alpha_{i}$ $(m^2)$', 
                          '$\\rho \\varphi_{hvac}$  (W)']
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    plt.rcParams['font.size'] = 20
    
    fig, axs = plt.subplots(5, 1, figsize=(11, 20))
    sensor_counts = sorted(optimization_results.keys(), reverse=False)  # Sorted in descending order

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

            # Create twin axis for density
            ax_twin = axs[param_index].twinx()
            
            sns.histplot(all_values, kde=False, ax=axs[param_index], stat="count", 
                         label=f"Sensor {sensor_count}", color=colors[sensor_idx % len(colors)], 
                         bins=100, alpha=0.5, edgecolor='black')
            sns.kdeplot(all_values, ax=ax_twin, color=colors[sensor_idx % len(colors)], linewidth=3)
            
            # Setting labels and limits
            axs[param_index].set_xlim([min(all_values), max(all_values)])
            ax_twin.set_xlim([min(all_values), max(all_values)])
            ax_twin.set_ylim(0, ax_twin.get_ylim()[1])  # Adjust y-limits for density
            
            # Hide density axis lines and ticks on right
            ax_twin.yaxis.set_visible(False)

        axs[param_index].set_xlabel(parameter_x_labels[param_index], fontsize=18)
        axs[param_index].set_ylabel('Counts', fontsize=18)
        axs[param_index].tick_params(axis='x', labelsize=18)
        axs[param_index].tick_params(axis='y', labelsize=18)
        
        if param_index == 0:
            axs[param_index].legend(title="Sensor Count", ncol=2, fontsize=14)
        else:
            axs[param_index].legend([], [], frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.23)  # Adjust horizontal space to prevent overlap
    plt.savefig('parameter_distribution.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.show()


plot_parameter_distributions(optimization_results_R2C2_decent)



  #%%


class R2C2Model:
    def __init__(self, Ri, Re, Ci, Ce, Ai, Ae, roxP_hvac):
        self.Ri = Ri
        self.Re = Re
        self.Ci = Ci
        self.Ce = Ce
        self.Ai = Ai
        self.Ae = Ae
        self.roxP_hvac = roxP_hvac
        self.N_states = 2
        self.update_matrices()

    def update_matrices(self):
        self.Ac = np.array([[-1/(self.Ci*self.Ri), 1/(self.Ci*self.Ri)],
                            [1/(self.Ce*self.Ri), -1/(self.Ce*self.Ri) - 1/(self.Ce*self.Re)]])
        self.Bc = np.array([[0, -self.roxP_hvac / self.Ci, self.Ai / self.Ci],
                            [1/(self.Ce*self.Re), 0, self.Ae/self.Ce]])
        self.Cc = np.array([[1, 0]])

    def discretize(self, dt):
        n = self.N_states
        F = expm(self.Ac * dt)
        G = np.dot(inv(self.Ac), np.dot(F - np.eye(n), self.Bc))
        H = self.Cc
        return F, G

    def predict(self, T, Te, T_ext, u, ghi, dt):
        F, G = self.discretize(dt)
        state_matrix = np.vstack((T, Te)).T
        input_matrix = np.vstack((T_ext, u, ghi)).T
        predictions = (F @ state_matrix.T) + (G @ input_matrix.T)
        predictions = predictions.T
        return predictions[:, 0], predictions[:, 1]  # Return predictions for T and Te separately

def time_for_temp_increase(model_params, initial_conditions, dt=300):
    model = R2C2Model(*model_params)
    T_init, Te_init, T_ext, ghi, duty_cycle = initial_conditions
    time_passed = 0
    T, Te = T_init, Te_init
    
    while True:
        T_pred, Te_pred = model.predict(np.array([T]), np.array([Te]), np.array([T_ext]), np.array([duty_cycle]), np.array([ghi]), dt)
        dT = T_pred - T_init
        print('T_pred', T_pred)
        print('dT', dT)
        if dT >= 1.1:
            print('temp limit reached')
            return time_passed
            
        T, Te = T_pred, Te_pred  # Update temperatures for the next prediction
        time_passed += dt / 60  # Convert time from seconds to minutes
        
model_params = (0.01, 0.01, 1000000, 1000000, 0.2, 0.2, 1000)  
initial_conditions = (295, 297, 300, 200, 0)  # Initial conditions (T_init, Te_init, T_ext, ghi, duty_cycle)
time_to_increase = time_for_temp_increase(model_params, initial_conditions)
print(f"Time to increase temperature by 1.1 K: {time_to_increase} minutes")


#%%
def make_predictions_for_all_models(optimization_results, T_initial, Te_initial, T_ext, u, ghi, delta_T=1.1):
    prediction_results = {}
    
    for sensor_count, houses in optimization_results.items():
        prediction_results[sensor_count] = {}
        
        for house_id, house_data in houses.items():
            models_predictions = []
            for i in range(sensor_count+1): 
                # Unpack parameters for each model
                Ri = house_data['optimal_params']['Ri'][i]
                Re = house_data['optimal_params']['Re'][i]
                Ci = house_data['optimal_params']['Ci'][i]
                Ce = house_data['optimal_params']['Ce'][i] 
                Ai = house_data['optimal_params']['Ai'][i]
                Ae = house_data['optimal_params']['Ae'][i]  
                roxP_hvac = house_data['optimal_params']['roxP_hvac'][i]

                model_params = (Ri, Re, Ci, Ce, Ai, Ae, roxP_hvac)
                initial_conditions = (T_initial, Te_initial, T_ext, ghi, u)
                
                # Calculate the time for the temperature to increase using the new model function
                time_to_increase = time_for_temp_increase(model_params, initial_conditions)
                models_predictions.append(time_to_increase)
            
            # Store predictions for each model in the house
            prediction_results[sensor_count][house_id] = models_predictions
    
    return prediction_results


T_initial = 295  # Initial indoor temperature in K
Te_initial = 300 # Initial envelope temperature in K
T_ext = 305  # External temperature in K
u = 0  # HVAC duty cycle
ghi = 1000  # Solar radiation in W/m2

# Call the function
predictions_all_models = make_predictions_for_all_models(optimization_results_R2C2_decent, T_initial, Te_initial, T_ext, u, ghi)

print(predictions_all_models)
#%%


def print_prediction_statistics(prediction_results):
    """
    Prints the statistics (mean, std, max, min, and median) of the prediction durations
    for each sensor group in the prediction_results dictionary.
    
    Parameters:
    prediction_results (dict): Dictionary containing the prediction durations for each sensor group and house.
    """
    for sensor_count, houses in prediction_results.items():
        durations = []
        # Collect all durations for the current sensor count
        for house_id, predictions in houses.items():
            durations.extend(predictions)
        
        # Convert durations to a numpy array for easy statistical computation
        durations = np.array(durations)
        
        # Calculate statistics
        mean_duration = np.mean(durations)
        std_duration = np.std(durations)
        max_duration = np.max(durations)
        min_duration = np.min(durations)
        median_duration = np.median(durations)
        
        # Print the statistics for the current sensor group
        print(f"Sensor Group {sensor_count}:")
        print(f"  Mean Duration: {mean_duration:.2f} mins")
        print(f"  Standard Deviation: {std_duration:.2f} mins")
        print(f"  Max Duration: {max_duration:.2f} mins")
        print(f"  Min Duration: {min_duration:.2f} mins")
        print(f"  Median Duration: {median_duration:.2f} mins\n")


print_prediction_statistics(predictions_all_models)
#%%

def calculate_duration_gaps_and_plot(predictions_all_models):
    # Prepare a list to collect the data
    data = []
    
    # Iterate through the predictions to calculate duration gaps
    for sensor_count, houses in predictions_all_models.items():
        for house_id, predictions in houses.items():
            if predictions:  # Ensure there are predictions to avoid errors
                max_duration = max(predictions)
                min_duration = min(predictions)
                duration_gap = max_duration - min_duration
                data.append({'sensor_count': sensor_count, 'duration_gap': duration_gap})

    # Convert the collected data into a DataFrame
    df = pd.DataFrame(data)

    # Plotting
    plt.figure(figsize=(10, 6))
    #plt.title('Duration Gap Distribution by Sensor Group')
    sns.boxplot(x='sensor_count', y='duration_gap', data=df, showfliers=False)
    plt.xlabel('Sensor Group')
    plt.ylabel('Duration Gap (minutes)')
    plt.grid(True)
    plt.savefig('duration_gap.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

    return df
df_gaps = calculate_duration_gaps_and_plot(predictions_all_models)
#%%
def sum_roxP_hvac_for_houses(optimization_results):
    """
    Sums the roxP_hvac values across sensors for each house in the optimization results.

    Parameters:
    - optimization_results: A dictionary with a nested structure [sensor_count][house_id] containing optimization outcomes.

    Returns:
    A dictionary with house_id as the key and the sum of roxP_hvac values as the value.
    """
    house_roxP_hvac_sums = {}
    for sensor_count, houses in optimization_results.items():
        for house_id, results in houses.items():
            if house_id not in house_roxP_hvac_sums:
                house_roxP_hvac_sums[house_id] = sum(results['optimal_params']['roxP_hvac'])
            else:
                house_roxP_hvac_sums[house_id] += sum(results['optimal_params']['roxP_hvac'])
    
    return house_roxP_hvac_sums

# Now, use the new function to compute the sums
house_roxP_hvac_sums = sum_roxP_hvac_for_houses(optimization_results_R2C2_decent)
print(house_roxP_hvac_sums)


def run_simulation(T0, Te0, T_ext, ghi, setpoint, model_params, simulation_hours=2, dt_minutes=5):
    """
    Runs a thermal model simulation with HVAC control, considering both indoor and envelope temperatures.

    Parameters:
    - T0 (float): Initial indoor temperature in Kelvin.
    - Te0 (float): Initial envelope temperature in Kelvin.
    - T_ext (float): External temperature in Kelvin (assumed constant for simplicity).
    - ghi (float): Solar radiation in W/m^2 (assumed constant).
    - setpoint (float): Temperature setpoint for HVAC control in Kelvin.
    - model_params (dict): Parameters for the R2C2Model.
    - simulation_hours (int): Duration of the simulation in hours.
    - dt_minutes (int): Time step for the simulation in minutes.

    Returns:
    - DataFrame: Simulation results including time, indoor temperature, envelope temperature, and HVAC duty cycle.
    """
    # Initialize the model
    model = R2C2Model(**model_params)

    # Convert simulation duration and time step to seconds
    dt = dt_minutes * 60
    total_time = simulation_hours * 60 * 60
    steps = int(total_time / dt)

    # Initialize simulation arrays
    times = np.arange(0, total_time, dt)
    indoor_temperatures = np.zeros(steps)
    envelope_temperatures = np.zeros(steps)
    duty_cycles = np.zeros(steps)
    indoor_temperatures[0], envelope_temperatures[0] = T0, Te0
    u = 0  # Initial HVAC duty cycle (off)

    # Simulation loop
    for i in range(1, steps):
        # Predict next temperatures for indoor and envelope
        T_pred, Te_pred = model.predict(indoor_temperatures[i-1], envelope_temperatures[i-1], T_ext, u, ghi, dt)
        
        indoor_temperatures[i], envelope_temperatures[i] = T_pred, Te_pred
        
        # HVAC control logic based on the indoor temperature
        if indoor_temperatures[i] > setpoint and u == 0:
            u = 1  # Turn HVAC on
        elif indoor_temperatures[i] <= setpoint - 1.1 and u == 1:
            u = 0  # Turn HVAC off
        duty_cycles[i] = u

    # Create and return the results DataFrame
    return pd.DataFrame({
        'Time (s)': times,
        'Indoor Temperature (K)': indoor_temperatures,
        'Envelope Temperature (K)': envelope_temperatures,
        'HVAC Duty Cycle': duty_cycles
    })




#%%
def calculate_energy_consumption(house_roxP_hvac_sums, optimization_results_R2C2_decent, T0, T_ext, ghi, setpoint, COP):
    energy_consumption_results = {}
    dt = 300  # Time step in seconds (5 minutes)

    for sensor_count, houses in optimization_results_R2C2_decent.items():
        energy_consumption_results[sensor_count] = {}
        
        for house_id, house_data in houses.items():
            house_energy_consumptions = []
            
            for i in range(len(house_data['optimal_params']['roxP_hvac'])):
                model_params = {
                    'Ri': house_data['optimal_params']['Ri'][i],
                    'Re': house_data['optimal_params']['Re'][i],
                    'Ci': house_data['optimal_params']['Ci'][i],
                    'Ai': house_data['optimal_params']['Ai'][i],
                    'Ce': house_data['optimal_params']['Ce'][i],
                    'Ae': house_data['optimal_params']['Ae'][i],
                    'roxP_hvac': house_data['optimal_params']['roxP_hvac'][i],
                }
                
                # Run simulation for each set of optimal parameters
                simulation_results = run_simulation(
                    T0=T0, 
                    Te0=(T0+T_ext)/2,
                    T_ext=T_ext,
                    ghi=ghi,
                    setpoint=setpoint,
                    model_params=model_params,
                    simulation_hours=2, 
                    dt_minutes=5
                )

                # Count how many times the HVAC was on
                hvac_on_times = simulation_results['HVAC Duty Cycle'].sum()
                
                # Compute energy consumption in Ws and convert to kWh 
                energy_consumption = (hvac_on_times * dt * (house_roxP_hvac_sums[house_id])/COP)/ 3600 /1000
                
                house_energy_consumptions.append(energy_consumption)
            
            energy_consumption_results[sensor_count][house_id] = house_energy_consumptions
    
    return energy_consumption_results


baseline_consumption_results = calculate_energy_consumption(house_roxP_hvac_sums, optimization_results_R2C2_decent, T0=295, T_ext=305, ghi=1000, setpoint=295, COP=2.5)

DR_consumption_results = calculate_energy_consumption(house_roxP_hvac_sums, optimization_results_R2C2_decent, T0=295, T_ext=305, ghi=1000, setpoint=296.1, COP=2.5)

#%%

def calculate_and_plot_flexibility(house_roxP_hvac_sums, optimization_results_R2C2_decent, T0, T_ext, ghi, setpoint_baseline, setpoint_DR, COP):
    # Initialize storage for summed flexibilities
    total_worst_flexibilities = None
    total_best_flexibilities = None
    total_thermostat_flexibilities = None

    # Aggregate data from houses with 3, 4, and 5 sensors
    for sensor_count in [ 4, 5]:
        houses = optimization_results_R2C2_decent.get(sensor_count, {})
        
        # Determine the number of steps for a 2-hour period with a 5-minute timestep
        steps = (2 * 60) // 5  # Plus one to include the final step

        for house_id, house_data in houses.items():
            room_flexibilities = []

            for i, _ in enumerate(house_data['optimal_params']['roxP_hvac']):
                model_params = {param: house_data['optimal_params'][param][i] for param in ['Ri', 'Re', 'Ci', 'Ai', 'Ce', 'Ae', 'roxP_hvac']}
                
                # Run simulations for baseline and DR
                baseline_results = run_simulation(T0, (T0 + T_ext) / 2, T_ext, ghi, setpoint_baseline, model_params)
                DR_results = run_simulation(T0, (T0 + T_ext) / 2, T_ext, ghi, setpoint_DR, model_params)
                
                # Calculate flexibility for this room/sensor, adjusting for timestep
                flexibility = np.append((baseline_results['HVAC Duty Cycle'][1:] - DR_results['HVAC Duty Cycle'][1:]) * (house_roxP_hvac_sums[house_id] / COP / 1000), 0)
                
                room_flexibilities.append(flexibility if len(flexibility) == steps else np.append(flexibility, 0))  # Ensure correct length

            worst_flexibility = np.min(room_flexibilities, axis=0)
            best_flexibility = np.max(room_flexibilities, axis=0)
            thermostat_flexibility = room_flexibilities[0]  

            # Summing flexibilities across all houses, ensuring arrays are initialized correctly
            if total_worst_flexibilities is None:
                total_worst_flexibilities, total_best_flexibilities, total_thermostat_flexibilities = worst_flexibility, best_flexibility, thermostat_flexibility
            else:
                total_worst_flexibilities += worst_flexibility
                total_best_flexibilities += best_flexibility
                total_thermostat_flexibilities += thermostat_flexibility

    # Convert timestep index to minutes, adjusting for the shift
    time_steps_minutes = np.arange(5, len(total_worst_flexibilities) * 5 + 5, 5)  # Start from 5 minutes

    # Compute the area under the thermostat curve in kWh
    thermostat_area = np.trapz(total_thermostat_flexibilities, dx=5 * 60) / 3600
    worst_gap_area = np.trapz(np.abs(total_worst_flexibilities - total_thermostat_flexibilities), dx=5 * 60) / 3600   # Convert W*seconds to kWh
    best_gap_area = np.trapz(np.abs(total_best_flexibilities - total_thermostat_flexibilities), dx=5 * 60) / 3600 
    # Plotting
    plt.figure(figsize=(12, 8))

    # Highlight the area under the thermostat curve
    plt.fill_between(time_steps_minutes, 0, total_thermostat_flexibilities, color='lightgrey', alpha=0.5)

    plt.fill_between(time_steps_minutes, total_worst_flexibilities, total_thermostat_flexibilities, color='red', alpha=0.3)
    plt.fill_between(time_steps_minutes, total_thermostat_flexibilities, total_best_flexibilities, color='green', alpha=0.3)
    
    plt.plot(time_steps_minutes, total_worst_flexibilities, label='Worst Performing Nodes', color='red')
    plt.plot(time_steps_minutes, total_thermostat_flexibilities, label='Thermostat', color='black', linewidth=3)
    plt.plot(time_steps_minutes, total_best_flexibilities, label='Best Performing Nodes', color='green')
    plt.annotate(f'-{worst_gap_area:.2f} kWh', xy=(0.8, 0.45), xycoords='axes fraction', ha='center', color='red', weight='bold',fontsize=20)
    plt.annotate(f'+{best_gap_area:.2f} kWh', xy=(0.8, 0.82), xycoords='axes fraction', ha='center', color='green', weight='bold',fontsize=20)

    plt.xlabel('Time (Minutes)', fontsize=20)
    plt.ylabel('Power Reduction (kW)', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc='lower left', fontsize=20)
    plt.savefig('power_red.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

    return thermostat_area
area=calculate_and_plot_flexibility(house_roxP_hvac_sums, optimization_results_R2C2_decent, 295, 305, 1000, 295, 296.1, 2.5)

#%%


def plot_phvac_distribution_with_normal_curve(optimization_results):
    # Determine the number of sensor groups for subplot arrangement
    num_sensor_groups = len(optimization_results)
    fig, axes = plt.subplots(num_sensor_groups, 1, figsize=(10, 4 * num_sensor_groups))

    for idx, (sensor_count, houses) in enumerate(optimization_results.items()):
        summation_phvacs = [sum(house['optimal_params']['roxP_hvac']) for house in houses.values()]

        # Histogram for the summed P_hvac values
        n, bins, patches = axes[idx].hist(summation_phvacs, bins=100, alpha=0.6, color='g', density=True, label=f'Sensor Group {sensor_count}')
        
        # Fit a normal distribution to the data
        mu, std = norm.fit(summation_phvacs)
        
        # Plot the normal distribution curve
        xmin, xmax = axes[idx].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        axes[idx].plot(x, p, 'k', linewidth=2, label='Normal Distribution Fit')
        
        title = f'Distribution of Summed P_hvac - Sensor Group {sensor_count}'
        axes[idx].set_title(title)
        axes[idx].set_xlabel('Summed P_hvac')
        axes[idx].set_ylabel('Density')
        axes[idx].grid(True)
        axes[idx].legend()

    plt.tight_layout()
    plt.show()


plot_phvac_distribution_with_normal_curve(optimization_results_R2C2_decent)
#%%


def plot_all_phvac_distributions_together(optimization_results):
    plt.figure(figsize=(12, 8))
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    sensor_counts = sorted(optimization_results.keys())
    
    all_summation_phvacs = []  # Collect all values to determine global min and max

    for sensor_count in sensor_counts:
        houses = optimization_results[sensor_count]
        summation_phvacs = [sum(house['optimal_params']['roxP_hvac']) for house in houses.values()]
        all_summation_phvacs.extend(summation_phvacs)  # Extend the list with the current group's values
        
        # Fit a normal distribution to the data: mu (mean), std (standard deviation)
        mu, std = norm.fit(summation_phvacs)

        # Plot histogram
        n, bins, patches = plt.hist(summation_phvacs, bins=100, alpha=0.6, color=colors[sensor_counts.index(sensor_count)], density=True, label=f'Sensor Group {sensor_count}', edgecolor='black')
        
        # Normal distribution curve
        xmin, xmax = min(all_summation_phvacs), max(all_summation_phvacs)  # Updated to use min/max from all data
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, color=colors[sensor_counts.index(sensor_count)], linewidth=2)

    plt.xlim(xmin, xmax)  # Set the x-axis limits based on the collected data
    #plt.title('$\\varphi_{hvac}$ Distribution (W)')
    plt.xlabel('$\\varphi_{hvac}$ (W)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.savefig('p_hvac.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

plot_all_phvac_distributions_together(optimization_results_R2C2_decent)
