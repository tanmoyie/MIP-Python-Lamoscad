"""
Script: run_milp_BnC_s4.2.6.py
Purpose: Automate computational experiments for Section 4.2 (findings) of the manuscript.
Author: <TANMOY DAS>
Revision Date: 11 July 2025
"""

# %% Imports & Configurations
import time
import pickle
import pandas as pd
from src.models.model_lamoscad_large_scale import build_model, solve_model
from src.solvers import branch_and_cut
from src.preprocessing import data_loader, preprocess_utils
from src.config import config_loader

#%%
instance_o = [200, 300, 400, 500, 750, 1000]
instance_st = [10, 20, 30, 50]
results = []

# data_o300_s30
for oil in [200, 300, 400, 500, 750, 1000]:  # instance_o
    for sta in [10, 20, 30, 50]:
        try:
            instance = f'{oil} x {sta}'
            print(f'\n\n Solving {instance} ====================================')
            with open(f"../data/large scale processed dataset/preprocessed_data_o{oil}_s{sta}.pkl", "rb") as f:
                d = pickle.load(f)

            OilSpills, Resources, demand_or, demand_ov = d["OilSpills"], d["Resources"], d["demand_or"], d["demand_ov"]
            Stations, coordinates_st, coordinates_spill = d["Stations"], d["coordinates_st"], d["coordinates_spill"]
            Sensitivity_R, v_o, v_o_n, eta_o = d["Sensitivity_R"], d["v_o"], d["v_o_n"], d["eta_o"]
            Distance, Distance_n, t_os, t_os_n = d["Distance"], d["Distance_n"], d["t_os"], d["t_os_n"]
            F_s, pn_sor, A_sr, Eff_sor = d["F_s"], d["pn_sor"], d["A_sr"], d["Eff_sor"]

            # Load config
            cfg = config_loader.load_config("../src/config/model_config.yaml")
            M, gamma = cfg["general"]["M"], cfg["general"]["gamma"]
            nQ, nS, nH, nUN = cfg["general"]["nQ"], cfg["general"]["nS"], cfg["general"]["nH"], cfg["general"]["nUN"]
            Vehicles, c_v, C_r = cfg["assets"]["vehicles"], cfg["assets"]["c_v"], cfg["assets"]["C_r"]
            Q_vr = {(v, r): cfg["assets"]["Q_vr"][v][r] for v in cfg["assets"]["Q_vr"] for r in
                    cfg["assets"]["Q_vr"][v]}
            L_p_or = {(o, r): cfg["L_p_or_values"]["c_i"] for o in OilSpills for r in ["c", "i"]}
            L_p_or.update({(o, r): float("inf") for o in OilSpills for r in ["m"]})
            n_vs = {(v, s): cfg["assets"]["n_vs_values"][v] for v in Vehicles for s in Stations}
            safety_buffer = 3

            spill_df = preprocess_utils.create_spill_dataframe(coordinates_spill, v_o, Sensitivity_R)
            station_df = preprocess_utils.create_station_dataframe(coordinates_st)
            DistanceMax = 10
            # NumberStMax = 5 if sta in [10, 20] else 10
            NumberStMax = 5 if sta in [10, 20] else 10
            if sta == 10: NumberStMax = 4
            elif sta == 20: NumberStMax = 7
            elif sta == 30: NumberStMax = 10
            elif sta == 50: NumberStMax = 15
            #%% ------------------------------------- MILP -------------------------------------
            print('Running MILP')
            W = [3.1, 2.5, 0.25, 0.0025, 0.025, 25, 0.5, 0.25]
            start_time_milp = time.time()
            model_1, x_s, y_os, z_sor, h_sov = build_model(Stations, OilSpills, Resources, Vehicles, W,
                                                               v_o_n, eta_o, t_os_n, gamma, M, demand_or, demand_ov, L_p_or,
                                                               NumberStMax, A_sr, nH, nUN, nQ, Q_vr, n_vs,
                                                               F_s, C_r, Eff_sor, pn_sor, c_v, Distance, DistanceMax, 'model_p')

            model_objectives, coverage_percentage, resource_stockpile_r, x_s1, y_os1, z_sor_lamoscad, h_sov_lamoscad, solution_values, \
                        num_var_constr = solve_model(model_1, x_s, y_os, z_sor, h_sov, OilSpills)
            runtime_milp = round(time.time() - start_time_milp, 2)
            print('num_var_constr', num_var_constr)
            print('runtime_milp', runtime_milp)
            #%% ------------------------------------- Branch and Cut -------------------------------------
            print('Running BnC')
            W = [3.1, 2.5, 0.25, 0.0025, 0.025, 25, 0.025, 0.25]
            start_time_BnC = time.time()
            best_sol, LB_final, UB_final, obj1_from_rmp = branch_and_cut.branch_and_cut_loop(OilSpills, Stations, Resources, Vehicles,
                                    A_sr, C_r, Eff_sor, Distance, F_s,  v_o_n, eta_o, t_os_n, pn_sor,
                                    demand_or, demand_ov, nQ, Q_vr, n_vs, L_p_or, M, gamma, W, NumberStMax,
                                    max_iters=1, tolerance=0.01, stable_iterations=3)
            runtime_BnC = time.time() - start_time_BnC
            # print("Facilities opened:", [s for s in best_sol["x"] if best_sol["x"][s] > 0.5])
            # print("Final LB:", LB_final)
            # print("Final UB:", UB_final)

            results.append({
                "instance": instance,
                "num_var_constr": num_var_constr,
                "model_objectives[0]": model_objectives[0],
                "runtime_milp": runtime_milp,
                "milp_obj1_from_mp": round(obj1_from_rmp, 2),
                "runtime_BnC": round(runtime_BnC, 2)
            })

        except KeyError as e:
            print(e)
df_BnC = pd.DataFrame(results)

with pd.ExcelWriter('../results/computational_findings_s4.2.xlsx',
                    engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    df_BnC.to_excel(writer, sheet_name='tab7. BnC_vs_MILP', index=False)

print('Run complete for Section 4.2.6')
