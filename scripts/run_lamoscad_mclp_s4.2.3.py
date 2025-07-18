"""
Script: run_lamoscad_mclp_s4.2.3.py
Section 4.2.3 and 4.2.4
Purpose: Automate computational experiments for Section 4.2 of the manuscript (B&C in separate script).
Author: <TANMOY DAS>
Revision Date: 11 July 2025
"""

# %% Imports & Configurations
from src.models.model_lamoscad import build_model, solve_model
from src.models.model_mclp import build_model_mclp, solve_model_mclp
from src.preprocessing import data_loader, preprocess_utils
from src.visualization.draw_network import draw_network_diagram
from src.config import config_loader
from src.utils import utility_functions
import pickle
import time
import pandas as pd

#%% Data Loading
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
W = [2.5, 2.5, 0.25, 0.0025, 0.025, 25, 0.25, 0.25]
M, gamma = cfg["general"]["M"], cfg["general"]["gamma"]
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

# %% ========================================== Start of Section 4.2.3 =================================================
# ------------------------------------------------- LAMOSCAD -----------------------------------------------------------
start_time_lamoscad = time.time()
model_1, x_s, y_os, z_sor, h_sov = build_model(Stations, OilSpills, Resources, Vehicles, W,
                                               v_o_n, eta_o, t_os_n, gamma, M, demand_or, demand_ov, L_p_or,
                                               NumberStMax, A_sr, nH, nUN, nQ, Q_vr, n_vs,
                                               F_s, C_r, Eff_sor, pn_sor, c_v, Distance, DistanceMax, 'model_p')

model_objectives, coverage_percentage_lamoscad, resource_stockpile_r, x_s1, y_os1, z_sor_lamoscad, h_sov_lamoscad, solution_values \
    = solve_model(model_1, x_s, y_os, z_sor, h_sov, OilSpills)
runtime_lamoscad = time.time() - start_time_lamoscad
mean_response_time_lamoscad = draw_network_diagram(y_os1, spill_df, station_df, name='model_p')
results_table5_lamoscad = preprocess_utils.convert_table5(
    coverage_percentage_lamoscad, mean_response_time_lamoscad, 25, runtime_lamoscad, len(x_s1), 'lamoscad')

# ------------------------------------------------- MCLP ---------------------------------------------------------------
model_mclp, x_s, y_os = build_model_mclp(OilSpills, Stations, v_o_n, NumberStMax, Distance, DistanceMax)
model_mclp, y_os1_mclp, coverage_percentage_mclp, num_sensitive_spills_mclp, runtime_mclp, number_facility_selected_mclp = solve_model_mclp(
    model_mclp, x_s, y_os, OilSpills)
mean_response_time_mclp = draw_network_diagram(y_os1_mclp, spill_df, station_df, name='model_mclp')
results_table5_mclp = preprocess_utils.convert_table5(coverage_percentage_mclp, mean_response_time_mclp, 21,
                                                      runtime_mclp, number_facility_selected_mclp,
                                                      'model_mclp')

# ------------------------------------------------- Table 5. LAMOSCAD vs MCLP ------------------------------------------
tab5_MCLP_vs_proposed_optimization_model = results_table5_lamoscad.merge(results_table5_mclp, on='Metric', how='left')
comparison = []
for i, row in tab5_MCLP_vs_proposed_optimization_model.iterrows():
    if i in [0, 2]:  # Percentage difference for 1st and 3rd row
        percent_diff = ((row["lamoscad"] - row["model_mclp"]) / row["model_mclp"]) * 100
        comparison.append(f"{percent_diff:.2f}%")
    else:  # Subtraction for other rows
        diff = row["model_mclp"] - row["lamoscad"]
        comparison.append(f"{diff:.2f}")
# Add the new column to the dataframe
tab5_MCLP_vs_proposed_optimization_model["Improvement"] = comparison

# ------------------------------------------------- Table 6. Resource allocation ---------------------------------------
safety_buffer = 3
tab6_resource_allocation = utility_functions.compute_stockpiling(z_sor_lamoscad, h_sov_lamoscad, safety_buffer)

# %% # Create Excel file from previously created dataframe
with pd.ExcelWriter('../results/4.2_computational_findings.xlsx',
                    engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    tab5_MCLP_vs_proposed_optimization_model.to_excel(writer, sheet_name='tab5. mclp vs lamoscad', index=False)
    tab6_resource_allocation.to_excel(writer, sheet_name='tab6. asset allocation', index=False)

print("Section 4.2.4 computational findings complete.")
