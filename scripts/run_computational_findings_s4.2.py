"""
Script: run_computational_findings_s4.2.py
Section 4.2.1, 4.2.2, 4.2.5
Purpose: Automate computational experiments for Section 4.2 of the manuscript (B&C in separate script).
Author: <TANMOY DAS>
Revision Date: 11 July 2025
"""

# %% 📦 Imports & Configurations
from src.models.model import build_model, solve_model
from src.preprocessing import data_loader, preprocess_utils
from src.visualization.draw_network import draw_network_diagram
from src.utils import utility_functions
from src.config import config_loader
import pickle
import time
import pandas as pd

# %% ======================================== Start of Table 4 =========================================================
""" Obtain Table 4, Fig 5,7 in this codeblock """
# Data Loading
with open("../data/preprocessed_data_model_p256.pkl", "rb") as f:
    data = pickle.load(f)

(OilSpills, Resources, demand_or, demand_ov, coordinates_spill, Sensitivity_R,
 v_o, v_o_n, eta_o, Stations, coordinates_st, A_sr, Eff_sor,
 Distance, Distance_n, t_os, F_s, t_os_n, pn_sor) = (
    data["OilSpills"], data["Resources"], data["demand_or"], data["demand_ov"], data["coordinates_spill"],
    data["Sensitivity_R"],
    data["v_o"], data["v_o_n"], data["eta_o"], data["Stations"], data["coordinates_st"], data["A_sr"], data["Eff_sor"],
    data["Distance"], data["Distance_n"], data["t_os"], data["F_s"], data["t_os_n"], data["pn_sor"])

# Load config parameters
cfg = config_loader.load_config("../src/config/model_config.yaml")
M, gamma = cfg["general"]["M"], cfg["general"]["gamma"]
# W = cfg["general"]["weights"]
W = [2.5, 2.5, 0.25, 0.0025, 0.025, 0.25, .75, 0.25]
nQ, nS, nH, nUN = cfg["general"]["nQ"], cfg["general"]["nS"], cfg["general"]["nH"], cfg["general"]["nUN"]
DistanceMax, NumberStMax = cfg["general"]["DistanceMax"], cfg["general"]["NumberStMax"]

# Asset parameters
Vehicles, c_v, C_r = cfg["assets"]["vehicles"], cfg["assets"]["c_v"], cfg["assets"]["C_r"]
Q_vr = {(v, r): cfg["assets"]["Q_vr"][v][r] for v in cfg["assets"]["Q_vr"] for r in cfg["assets"]["Q_vr"][v]}
n_vs = {(v, s): cfg["assets"]["n_vs_values"][v] for v in Vehicles for s in Stations}

L_p_or = {(o, r): cfg["L_p_or_values"]["c_i"] for o in OilSpills for r in ["c", "i"]}
L_p_or.update({(o, r): float("inf") for o in OilSpills for r in ["m"]})
spill_df = preprocess_utils.create_spill_dataframe(coordinates_spill, v_o, Sensitivity_R)
station_df = preprocess_utils.create_station_dataframe(coordinates_st)

# %% Experimental Design
model_config = ['model_2', 'model_p', 'model_5', 'model_6', 'model_c', 'model_3']  # ,
NumberStMax_dict = {'model_c': 4, 'model_2': 4, 'model_3': 5, 'model_p': 5, 'model_5': 6, 'model_6': 8}
DistanceMax_dict = {'model_c': 15, 'model_2': 10, 'model_3': 10, 'model_p': 10, 'model_5': 10, 'model_6': 10}
Objective1_dict = {m: None for m in model_config}
Objective2_dict = {m: None for m in model_config}
Coverage_dict = {m: None for m in model_config}
Mean_response_time_dict = {m: None for m in model_config}

# %% Run Optimization Models
# Model 2, Proposed, 5, 6
print("Running optimization models ")
for m1 in ['model_p', 'model_2', 'model_5', 'model_6']:
    print(f'Running {m1}')
    time_start_gurobi = time.time()
    model_1, x_s, y_os, z_sor, h_sov = build_model(Stations, OilSpills, Resources, Vehicles, W,
                                                   v_o_n, eta_o, t_os_n, gamma, M, demand_or, demand_ov, L_p_or,
                                                   NumberStMax_dict[m1], A_sr, nH, nUN, nQ, Q_vr, n_vs,
                                                   F_s, C_r, Eff_sor, pn_sor, c_v, Distance, DistanceMax_dict[m1], m1)

    model_objectives, coverage_percentage, resource_stockpile_r, x_s1, y_os1, z_sor1, h_sov1, solution_values \
        = solve_model(model_1, x_s, y_os, z_sor, h_sov, OilSpills)
    # Draw the network fig 5(b,d) fig 7(a,b)
    # draw_network_diagram(y_os1, spill_df, station_df, name=m1)

    mean_response_time = utility_functions.compute_mean_response_time(y_os1, spill_df, station_df)
    Objective1_dict[m1] = model_objectives[0]
    Objective2_dict[m1] = model_objectives[1]
    Coverage_dict[m1] = coverage_percentage
    Mean_response_time_dict[m1] = mean_response_time

#%% Model 3
for m2 in ['model_3']:
    print(f'Running {m2}')
    # station data is different in model 3
    with open(f"../data/preprocessed_data_{m2}.pkl", "rb") as f:
        data = pickle.load(f)
    (Stations, coordinates_st, A_sr, Eff_sor, Distance, Distance_n,
     t_os, F_s, t_os_n, pn_sor) = (data["Stations"], data["coordinates_st"], data["A_sr"], data["Eff_sor"],
                                   data["Distance"], data["Distance_n"], data["t_os"], data["F_s"],
                                   data["t_os_n"], data["pn_sor"])
    n_vs = {(v, s): cfg["assets"]["n_vs_values"][v] for v in Vehicles for s in Stations}
    L_p_or = {(o, r): cfg["L_p_or_values"]["c_i"] for o in OilSpills for r in ["c", "i"]}
    L_p_or.update({(o, r): float("inf") for o in OilSpills for r in ["m"]})
    station_df = preprocess_utils.create_station_dataframe(coordinates_st)

    # Solve Model
    model_1, x_s, y_os, z_sor, h_sov = build_model(Stations, OilSpills, Resources, Vehicles, W,
                                                   v_o_n, eta_o, t_os, gamma, M, demand_or, demand_ov, L_p_or,
                                                   NumberStMax_dict[m2], A_sr, nH, nUN, nQ, Q_vr, n_vs,
                                                   F_s, C_r, Eff_sor, pn_sor, c_v, Distance, DistanceMax_dict[m2], m2)
    model_objectives, coverage_percentage, resource_stockpile_r, x_s1, y_os1, z_sor1, h_sov1, solution_values \
        = solve_model(model_1, x_s, y_os, z_sor, h_sov, OilSpills)
    # draw_network_diagram(y_os1, spill_df, station_df, name=m2)

    mean_response_time = utility_functions.compute_mean_response_time(y_os1, spill_df, station_df)
    Objective1_dict[m2] = model_objectives[0]
    Objective2_dict[m2] = model_objectives[1]
    Coverage_dict[m2] = coverage_percentage
    Mean_response_time_dict[m2] = mean_response_time

#%%  Current (Model C) — Facility Location Variants
W[6] = 0.6
for m2 in ['model_c']:
    print(f'Running {m2}')
    # station data is different in model_c
    with open(f"../data/preprocessed_data_{m2}.pkl", "rb") as f:
        data = pickle.load(f)
    (Stations, coordinates_st, A_sr, Eff_sor, Distance, Distance_n,
     t_os, F_s, t_os_n, pn_sor) = (data["Stations"], data["coordinates_st"], data["A_sr"], data["Eff_sor"],
                                   data["Distance"], data["Distance_n"], data["t_os"], data["F_s"],
                                   data["t_os_n"], data["pn_sor"])
    n_vs = {(v, s): cfg["assets"]["n_vs_values"][v] for v in Vehicles for s in Stations}
    L_p_or = {(o, r): cfg["L_p_or_values"]["c_i"] for o in OilSpills for r in ["c", "i"]}
    L_p_or.update({(o, r): float("inf") for o in OilSpills for r in ["m"]})
    station_df = preprocess_utils.create_station_dataframe(coordinates_st)

    # Solve Model
    model_1, x_s, y_os, z_sor, h_sov = build_model(Stations, OilSpills, Resources, Vehicles, W,
                                                   v_o_n, eta_o, t_os, gamma, M, demand_or, demand_ov, L_p_or,
                                                   NumberStMax_dict[m2], A_sr, nH, nUN, nQ, Q_vr, n_vs,
                                                   F_s, C_r, Eff_sor, pn_sor, c_v, Distance, DistanceMax_dict[m2], m2)
    model_objectives, coverage_percentage, resource_stockpile_r, x_s1, y_os1, z_sor1, h_sov1, solution_values \
        = solve_model(model_1, x_s, y_os, z_sor, h_sov, OilSpills)
    # draw_network_diagram(y_os1, spill_df, station_df, name=m2)

    mean_response_time = utility_functions.compute_mean_response_time(y_os1, spill_df, station_df,  modelType=True)
    Objective1_dict[m2] = model_objectives[0]
    Objective2_dict[m2] = model_objectives[1]
    Coverage_dict[m2] = coverage_percentage
    Mean_response_time_dict[m2] = mean_response_time

# %% Generate Table 4: Models experimental design
model_config = ['model_c', 'model_2', 'model_3', 'model_p', 'model_5', 'model_6']  #  ,
data = {
    'Model config': model_config,
    'N': [NumberStMax_dict[m] for m in model_config],
    'Max Distance (in Km)': [80 * DistanceMax_dict[m] for m in model_config],  # 80 to convert GIS data into km
    'Objective 1': [Objective1_dict[m] for m in model_config],
    'Objective 2': [round(Objective2_dict[m] / 1000, 2) for m in model_config],
    'Coverage Percentage (in %)': [f'{Coverage_dict[m]}%' for m in model_config],
    'Min Response Time (in hr)': [Mean_response_time_dict[m] for m in model_config]}

df_table4 = pd.DataFrame(data, index=model_config)

# %% # Create Excel file from previously created dataframe
with pd.ExcelWriter('../results/computational_findings_s4.2.xlsx',
                    engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    df_table4.to_excel(writer, sheet_name='tab4. experimental design', index=False)
print(df_table4[['Model config','Objective 1', 'Objective 2']])
print("Section 4.2.5 computational findings complete.")