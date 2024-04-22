
#%%
from preprocessing import read_csvs_to_dfs, process_house_data, print_optimization_statistics, plot_error_distribution
from sklearn.metrics import mean_squared_error

# Main output directory name
main_output_directory = "house_data_csvs1"

# Read the CSVs back into dictionaries
all_houses_reduced = read_csvs_to_dfs(main_output_directory)
#%%
# Apply the processing function to all houses in all_houses_reduced, skipping those without 'GHI'
processed_houses_reduced = {}
for sensor_count, houses in all_houses_reduced.items():
    processed_houses = {}
    for house_id, df in houses.items():
        if 'GHI' not in df.columns:
            print(f"Skipping house {house_id} due to missing 'GHI' column.")
            continue  # Skip this house
        processed_houses[house_id] = process_house_data(df.copy())
    processed_houses_reduced[sensor_count] = processed_houses

#%% Training of R2C2Decent
from Decent2R2C import fit_models_R2C2Decent, plot_parameter_distributions_R2C2Decent, save_results_R2C2Decent

optimization_results_R2C2_decent=fit_models_R2C2Decent(processed_houses_reduced)

print_optimization_statistics(optimization_results_R2C2_decent)

plot_error_distribution(optimization_results_R2C2_decent)

plot_parameter_distributions_R2C2Decent(optimization_results_R2C2_decent)

save_results_R2C2Decent(optimization_results_R2C2_decent)

#%% Training of R1C1Decent
from Decent1R1C import fit_models_R1C1Decent, plot_parameter_distributions_R1C1Decent, save_results_R1C1Decent

optimization_results_R1C1_decent=fit_models_R1C1Decent(processed_houses_reduced)

print_optimization_statistics(optimization_results_R1C1_decent)

plot_error_distribution(optimization_results_R1C1_decent)

plot_parameter_distributions_R1C1Decent(optimization_results_R1C1_decent)

save_results_R1C1Decent(optimization_results_R1C1_decent)
#%% Training of R1C1Cent
from Cent1R1C import fit_models_R1C1Cent, plot_parameter_distributions_R1C1Cent, save_results_R1C1Cent

optimization_results_R1C1_cent=fit_models_R1C1Cent(processed_houses_reduced)

print_optimization_statistics(optimization_results_R1C1_cent)

plot_error_distribution(optimization_results_R1C1_cent)

plot_parameter_distributions_R1C1Cent(optimization_results_R1C1_cent)

save_results_R1C1Cent(optimization_results_R1C1_cent)
#%% Training of R2C2Cent
from Cent2R2C import fit_models_R2C2Cent, save_results_R2C2Cent

optimization_results_R2C2_cent=fit_models_R2C2Cent(processed_houses_reduced)

print_optimization_statistics(optimization_results_R2C2_cent)

plot_error_distribution(optimization_results_R2C2_cent)

save_results_R2C2Cent(optimization_results_R2C2_cent)
#%% To analyze the parameters that were fit at the upper and lower bounds 
from filtering import collect_and_save_at_bounds, enrich_data
collect_and_save_at_bounds(optimization_results_R2C2_decent)

# Define the upper bounds for Ri and Re
ri_upper_bound = 0.05
re_upper_bound = 0.1

enrich_data('Ri_at_bounds.csv', all_houses_reduced, ri_upper_bound, re_upper_bound)

