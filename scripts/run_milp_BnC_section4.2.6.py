"""
Script: run_computational_findings_section4.2.py
Purpose: Automate computational experiments for Section 4.2 (findings) of the manuscript.
Author: <TANMOY DAS>
Revision Date: 11 July 2025
"""

# %% 📦 Imports & Configurations
from src.models.model_lamoscad_large_scale import build_model, solve_model
from src.models.model_mclp import build_model_mclp, solve_model_mclp
from src.solvers import branch_and_cut
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
OilSpills = data["OilSpills"]
Resources = data["Resources"]
demand_or = data["demand_or"]
demand_ov = data["demand_ov"]
coordinates_spill = data["coordinates_spill"]
Sensitivity_R = data["Sensitivity_R"]
v_o = data["v_o"]
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
t_os_n = data["t_os_n"]
pn_sor = data["pn_sor"]

# Load config
config = config_loader.load_config("../src/config/model_config.yaml")
# Extract scalar parameters
M = config["general"]["M"]
gamma = config["general"]["gamma"]
safety_buffer = 3
W = [2.5, 2.5, 0.25, 0.0025, 0.025, 25, 0.25, 0.25]  # why new weight values??
# W = config["general"]["weights"]
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
C_r = config["assets"]["C_r"]
n_vs = {(v, s): config["assets"]["n_vs_values"][v]
        for v in Vehicles for s in Stations}
L_p_or = {(o, r): config["L_p_or_values"]["c_i"] for o in OilSpills for r in ["c", "i"]}
L_p_or.update({(o, r): float("inf") for o in OilSpills for r in ["m"]})

spill_df = preprocess_utils.create_spill_dataframe(coordinates_spill, v_o, Sensitivity_R)
station_df = preprocess_utils.create_station_dataframe(coordinates_st)
#%% ------------------------------------- MILP -------------------------------------
start_time_milp = time.time()
model_1, x_s, y_os, z_sor, h_sov = build_model(Stations, OilSpills, Resources, Vehicles, W,
                                                   v_o_n, eta_o, t_os_n, gamma, M, demand_or, demand_ov, L_p_or,
                                                   NumberStMax, A_sr, nH, nUN, nQ, Q_vr, n_vs,
                                                   F_s, C_r, Eff_sor, pn_sor, c_v, Distance, DistanceMax, 'model_p')

model_objectives, coverage_percentage, resource_stockpile_r, x_s1, y_os1, z_sor_lamoscad, h_sov_lamoscad, solution_values, \
            num_var_constr, milp_gap = solve_model(model_1, x_s, y_os, z_sor, h_sov, OilSpills)
runtime_milp = round(time.time() - start_time_milp, 2)
resource_allocation = utility_functions.compute_stockpiling(z_sor_lamoscad, h_sov_lamoscad, safety_buffer)


#%% ------------------------------------- Branch and Cut -------------------------------------
start_time_BnC = time.time()
best_sol, LB_final, UB_final, milp_obj1_from_mp = branch_and_cut.branch_and_cut_loop(OilSpills, Stations, Resources, Vehicles,
                        A_sr, C_r, Eff_sor, Distance, F_s,  v_o_n, eta_o, t_os_n, pn_sor,
                        demand_or, demand_ov, nQ, Q_vr, n_vs, L_p_or, M, gamma, W, NumberStMax,
                        max_iters = 50, tolerance = 0.01, stable_iterations = 3)
runtime_BnC = time.time() - start_time_BnC
print("Facilities opened:", [s for s in best_sol["x"] if best_sol["x"][s] > 0.5])
print("Final LB:", LB_final)
print("Final UB:", UB_final)

instance = f'{len(OilSpills)} x {NumberStMax}'
df = pd.DataFrame({
    "instance": instance,
    "num_var_constr": num_var_constr,
    "model_objectives[0]": model_objectives[0],
    "runtime_milp": runtime_milp,
    "milp_gap": milp_gap,
    "milp_obj1_from_mp": milp_obj1_from_mp,
    "runtime_BnC": runtime_BnC
})

print(df)


#%%
# # Create Excel file for 5 previously created dataframe
# # df_table4.to_excel('../results/computational_findings_s4.2.xlsx', sheet_name='tab4. experimental design', index=False)
# with pd.ExcelWriter('../results/computational_findings_s4.2.xlsx', engine='openpyxl', mode='w') as writer:
#     df_table4.to_excel(writer, sheet_name='tab4. experimental design', index=False)
#     Table5_MCLP_vs_proposed_optimization_model.to_excel(writer, sheet_name='tab5. mclp vs lamoscad', index=False)
#     resource_allocation.to_excel(writer, sheet_name='tab6. asset allocation', index=False)
#

print("Section 4.2 computational findings complete.")