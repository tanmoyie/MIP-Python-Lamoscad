"""
Script: run_computational_findings_section4.2.py
Purpose: Automate computational experiments for Section 4.2 (findings) of the manuscript.
Author: <TANMOY DAS>
Revision Date: 11 July 2025
"""

# %% 📦 Imports & Configurations
from src.models import model_lamoscad, model_mclp  # Example
from src.models.model_lamoscad import build_model, solve_model
from src.models.model_mclp import build_model_mclp, solve_model_mclp
from src.preprocessing import data_loader, preprocess_utils
from src.visualization.draw_network import draw_network_diagram
from src.config import config_loader
from src.utils import utility_functions
import pickle
import time
import pandas as pd

# Load the preprocessed data
with open("../data/preprocessed_data_model_p256.pkl", "rb") as f:
    data = pickle.load(f)

# %% 📊 Data Loading
v_o = data["v_o_n"]
OilSpills = data["OilSpills"]
Resources = data["Resources"]
demand_or = data["demand_or"]
demand_ov = data["demand_ov"]
coordinates_spill = data["coordinates_spill"]
SizeSpill = data["SizeSpill"]
Sensitivity_R = data["Sensitivity_R"]
v_o_n = data["v_o_n"]
eta_o = data["eta_o"]

Stations = data["Stations"]
coordinates_st = data["coordinates_st"]
A_sr = data["A_sr"]
Eff_sor = data["Eff_sor"]
Distance = data["Distance"]
Distance_n = data["Distance_n"]
t_os = data["t_os"]
F_s = data["F_s"]
C_sr = data["C_sr"]
t_os_n = data["t_os_n"]
pn_sor = data["pn_sor"]

# Load config
config = config_loader.load_config("../src/config/model_config.yaml")
# Extract scalar parameters
M = config["general"]["M"]
gamma = config["general"]["gamma"]
W = config["general"]["weights"]
nQ = config["general"]["nQ"]
DistanceMax = config["general"]["DistanceMax"]
NumberStMax = config["general"]["NumberStMax"]
nS = config["general"]["nS"]
nH = config["general"]["nH"]
nUN = config["general"]["nUN"]
Vehicles = config["assets"]["vehicles"]
c_v = config["assets"]["c_v"]
Q_vr = {(vehicle, resource): config["assets"]["Q_vr"][vehicle][resource]
        for vehicle in config["assets"]["Q_vr"] for resource in config["assets"]["Q_vr"][vehicle]}

n_vs = {(v, s): config["assets"]["n_vs_values"][v]
        for v in Vehicles for s in Stations}

L_p_or = {(o, r): config["L_p_or_values"]["c_i"] for o in OilSpills for r in ["c", "i"]}
L_p_or.update({(o, r): float("inf") for o in OilSpills for r in ["m"]})
spill_df = preprocess_utils.create_spill_dataframe(coordinates_spill, SizeSpill, Sensitivity_R)
station_df = preprocess_utils.create_station_dataframe(coordinates_st)

# %% Experimental Design
model_config = ['model_2', 'model_p', 'model_5', 'model_6', 'model_c', 'model_3']  # ,
NumberStMax_dict = {'model_c': 4, 'model_2': 4, 'model_3': 5, 'model_p': 5, 'model_5': 6, 'model_6': 8}
DistanceMax_dict = {'model_c': 15, 'model_2': 10, 'model_3': 10, 'model_p': 10, 'model_5': 10, 'model_6': 10}
Objective1_dict = {m: None for m in model_config}
Objective2_dict = {m: None for m in model_config}
Coverage_dict = {m: None for m in model_config}
Mean_response_time_dict = {m: None for m in model_config}

# %% 🔧 Run Optimization Models
# Model 2, Proposed, 5, 6
print("Running optimization models ")
for m in ['model_2', 'model_p', 'model_5', 'model_6']:
    print(f'Running {m}')
    time_start_gurobi = time.time()
    model_1, x_s, y_os, z_sor, h_sov = build_model(Stations, OilSpills, Resources, Vehicles, W,
                                                   v_o, eta_o, t_os, gamma, M, demand_or, demand_ov, L_p_or,
                                                   NumberStMax_dict[m], A_sr, nH, nUN, nQ, Q_vr, n_vs,
                                                   F_s, C_sr, Eff_sor, pn_sor, c_v, Distance, DistanceMax_dict[m], m)
    model_objectives, coverage_percentage, resource_stockpile_r, x_s1, y_os1, z_sor1, h_sov1, solution_values \
        = solve_model(model_1, x_s, y_os, z_sor, h_sov, OilSpills)
    # Draw the network
    mean_response_time = draw_network_diagram(y_os1, spill_df, station_df, name=m)
    if m == 'model_p':
        results_table5_lamoscad = preprocess_utils.convert_table5(
            coverage_percentage, mean_response_time, 25, time.time()-time_start_gurobi, len(x_s1), 'lamoscad')
        z_sor_lamoscad = z_sor1
        h_sov_lamoscad = h_sov1
    # fig 5(b,d) fig 7(a,b)
    if m == 'model_p': z_sor1_modelP = z_sor1
    Objective1_dict[m] = model_objectives[0]
    Objective2_dict[m] = model_objectives[1]
    Coverage_dict[m] = coverage_percentage
    Mean_response_time_dict[m] = mean_response_time

# %% Model 3 and current
# Model 3 and Current (Model C) — Facility Location Variants
for m in ['model_c', 'model_3']:
    print(f'Running {m}')
    # Load the correct preprocessed dataset
    with open(f"../data/preprocessed_data_{m}.pkl", "rb") as f:
        data = pickle.load(f)

    Stations = data["Stations"]
    coordinates_st = data["coordinates_st"]
    A_sr = data["A_sr"]
    Eff_sor = data["Eff_sor"]
    Distance = data["Distance"]
    Distance_n = data["Distance_n"]
    t_os = data["t_os"]
    F_s = data["F_s"]
    C_sr = data["C_sr"]
    t_os_n = data["t_os_n"]
    pn_sor = data["pn_sor"]

    n_vs = {(v, s): config["assets"]["n_vs_values"][v] for v in Vehicles for s in Stations}
    L_p_or = {(o, r): config["L_p_or_values"]["c_i"] for o in OilSpills for r in ["c", "i"]}
    L_p_or.update({(o, r): float("inf") for o in OilSpills for r in ["m"]})

    station_df = preprocess_utils.create_station_dataframe(coordinates_st)

    # Solve Model
    time_start_gurobi = time.time()
    model_1, x_s, y_os, z_sor, h_sov = build_model(Stations, OilSpills, Resources, Vehicles, W,
                                                   v_o, eta_o, t_os, gamma, M, demand_or, demand_ov, L_p_or,
                                                   NumberStMax_dict[m], A_sr, nH, nUN, nQ, Q_vr, n_vs,
                                                   F_s, C_sr, Eff_sor, pn_sor, c_v, Distance, DistanceMax_dict[m], m)
    model_objectives, coverage_percentage, resource_stockpile_r, x_s1, y_os1, z_sor1, h_sov1, solution_values = solve_model(
        model_1, x_s, y_os, z_sor, h_sov, OilSpills)

    mean_response_time = draw_network_diagram(y_os1, spill_df, station_df, name=m)
    Objective1_dict[m] = model_objectives[0]
    Objective2_dict[m] = model_objectives[1]
    Coverage_dict[m] = coverage_percentage
    Mean_response_time_dict[m] = mean_response_time

# %% 📈 Generate Tables
# %% Table 4. Models experimental design
# Create a DataFrame
model_config = ['model_c', 'model_2', 'model_3', 'model_p', 'model_5', 'model_6']
data = {
    'Model config': model_config,
    'N': [NumberStMax_dict[m] for m in model_config],
    'Max Distance (in Km)': [80 * DistanceMax_dict[m] for m in model_config],  # 80 to convert GIS data into km
    'Objective 1': [Objective1_dict[m] for m in model_config],
    'Objective 2 (in thousands)': [round(Objective2_dict[m] / 1000, 2) for m in model_config],
    'Coverage Percentage (in %)': [f'{Coverage_dict[m]}%' for m in model_config],
    'Min Response Time (in hr)': [round(Mean_response_time_dict[m], 2) for m in model_config]}

df_table4 = pd.DataFrame(data, index=model_config)

# %% Table 5
model_mclp, x_s, y_os = build_model_mclp(OilSpills, Stations, v_o_n, NumberStMax, Distance, DistanceMax)
model_mclp, y_os1_mclp, coverage_percentage_mclp, num_sensitive_spills_mclp, runtime_mclp, number_facility_selected_mclp = solve_model_mclp(
    model_mclp, x_s, y_os, OilSpills)

mean_response_time_mclp = draw_network_diagram(y_os1_mclp, spill_df, station_df, name='model_mclp')

results_table5_mclp = preprocess_utils.convert_table5(coverage_percentage_mclp, mean_response_time_mclp, 21,
                                                               runtime_mclp, number_facility_selected_mclp,
                                                               'model_mclp')

Table5_MCLP_vs_proposed_optimization_model = results_table5_lamoscad.merge(results_table5_mclp, on='Metric', how='left')
comparison = []
for i, row in Table5_MCLP_vs_proposed_optimization_model.iterrows():
    if i in [0, 2]:  # Percentage difference for 1st and 3rd row
        percent_diff = ((row["lamoscad"] - row["model_mclp"]) / row["model_mclp"]) * 100
        comparison.append(f"{percent_diff:.2f}%")
    else:  # Subtraction for other rows
        diff = row["model_mclp"] - row["lamoscad"]
        comparison.append(f"{diff:.2f}")

# Add the new column to the dataframe
Table5_MCLP_vs_proposed_optimization_model["Comparison"] = comparison

# %% Figure 6

# %% Table 6
safety_buffer = 3

resource_allocation = utility_functions.compute_stockpiling(z_sor_lamoscad, h_sov_lamoscad, safety_buffer)

#%%
# Create Excel file for 5 previously created dataframe
# df_table4.to_excel('../results/computational_findings_s4.2.xlsx', sheet_name='tab4. experimental design', index=False)
with pd.ExcelWriter('../results/computational_findings_s4.2.xlsx', engine='openpyxl', mode='w') as writer:
    df_table4.to_excel(writer, sheet_name='tab4. experimental design', index=False)
    Table5_MCLP_vs_proposed_optimization_model.to_excel(writer, sheet_name='tab5. mclp vs lamoscad', index=False)
    resource_allocation.to_excel(writer, sheet_name='tab6. asset allocation', index=False)

print("Section 4.2 computational findings complete.")
