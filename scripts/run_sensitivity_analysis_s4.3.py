"""
Section 4.3

"""
from src.models.model_lamoscad import build_model, solve_model
from src.models.model_mclp import build_model_mclp, solve_model_mclp
from src.solvers import branch_and_cut
from src.preprocessing import data_loader, preprocess_utils
from src.visualization.draw_network import draw_network_diagram
from src.visualization.draw_parallel_coordinate_plot import draw_parallel_coordinate_plot
from src.visualization.draw_boxplot import draw_boxplot
from src import config
from src.config import config_loader
from src.utils import utility_functions
import pickle
import pandas as pd
import itertools
import random

#%% -------------------------------------------- Section 4.3.1 Impact of number of facilities --------------------------
# results = []
#
# for oil in [200, 100]:
#     sta = 20
#     # Load data for this oil size
#     with open(f"../data/large scale processed dataset/preprocessed_data_o{oil}_s{sta}.pkl", "rb") as f:
#         data = pickle.load(f)
#
#     OilSpills, Resources, demand_or, demand_ov = data["OilSpills"], data["Resources"], data["demand_or"], data["demand_ov"]
#     coordinates_spill, Sensitivity_R = data["coordinates_spill"], data["Sensitivity_R"]
#     v_o, v_o_n, eta_o = data["v_o"], data["v_o_n"], data["eta_o"]
#     Stations, coordinates_st, A_sr, Eff_sor = data["Stations"], data["coordinates_st"], data["A_sr"], data["Eff_sor"]
#     Distance, Distance_n, t_os, F_s, t_os_n, pn_sor = data["Distance"], data["Distance_n"], data["t_os"], data["F_s"], data["t_os_n"], data["pn_sor"]
#
#     cfg = config_loader.load_config("../src/config/model_config.yaml")
#     W, M, gamma = cfg["general"]["weights"], cfg["general"]["M"], cfg["general"]["gamma"]
#     DistanceMax = cfg["general"]["DistanceMax"]
#     nH, nUN, nQ = cfg["general"]["nH"], cfg["general"]["nUN"], cfg["general"]["nQ"]
#     Vehicles, c_v, C_r = cfg["assets"]["vehicles"], cfg["assets"]["c_v"], cfg["assets"]["C_r"]
#     Q_vr = {(v, r): cfg["assets"]["Q_vr"][v][r] for v in cfg["assets"]["Q_vr"] for r in cfg["assets"]["Q_vr"][v]}
#     n_vs = {(v, s): cfg["assets"]["n_vs_values"][v] for v in Vehicles for s in Stations}
#     L_p_or = {(o, r): cfg["L_p_or_values"]["c_i"] for o in OilSpills for r in ["c", "i"]}
#     L_p_or.update({(o, r): float("inf") for o in OilSpills for r in ["m"]})
#
#     spill_df = preprocess_utils.create_spill_dataframe(coordinates_spill, v_o, Sensitivity_R)
#     station_df = preprocess_utils.create_station_dataframe(coordinates_st)

#     for NumberStMax in [12, 8]:  #, 5, 4, 2]:  # ++ 15,
#         model_1, x_s, y_os, z_sor, h_sov = build_model(
#             Stations, OilSpills, Resources, Vehicles, W,
#             v_o_n, eta_o, t_os_n, gamma, M, demand_or, demand_ov, L_p_or,
#             NumberStMax, A_sr, nH, nUN, nQ, Q_vr, n_vs,
#             F_s, C_r, Eff_sor, pn_sor, c_v, Distance, DistanceMax, 'model_p')
#
#         model_objectives, coverage_pct, _, x_s1, y_os1, _, _, _ = solve_model(
#             model_1, x_s, y_os, z_sor, h_sov, OilSpills)
#
#         mean_response_time = utility_functions.compute_mean_response_time(y_os1, spill_df, station_df)
#         # open_facilities = [s for s in x_s1 if x_s1[s] >= 0.9]
#         results.append({
#             "Max_Facilities": NumberStMax,
#             "Num_Spills": len(OilSpills),
#             "Open_Facilities": list(x_s1.index),
#             "Coverage_%": coverage_pct,
#             "Mean_Response_Time": mean_response_time,
#         })
#
# # Save to Excel
# tab8_num_facilities = pd.DataFrame(results)
# df_results.to_excel("../results/facility_sensitivity_results.xlsx", index=False)


# %% ------------------------------------------ Section 4.3.2 Weight and sensitive oil spills  -------------------------
# w1_list = [.0025, 0.1, 0.2, .25, 0.5, 0.8, 1, 2.5, 100]
# w2_list = [.0025, 0.1, 0.2, .25, 0.5, 0.8, 1, 2.5, 100]
# w3_list = [.0025, 0.1, 0.2, .25, 0.5, 0.8, 1, 2.5, 100]
# w4_list = [0.25e-7, .0025, 0.01, 0.09, 0.1, 0.2, 1]
# w5_list = [0.0001, 0.001, 0.25, 0.5, 0.75, 1]
# w6_list = [0.0025, 0.25, 0.5, 0.75, 1, 100]
# w7_list = [0.0025, 0.025, 0.25, 0.5, 0.75, 1]
# w8_list = [0.001, 0.25, 0.5, 0.75, 1, 100]
#
# # Random Sampling Setup
# all_combinations = list(itertools.product(w1_list, w2_list, w3_list, w4_list,
#     w5_list, w6_list, w7_list, w8_list))
# sample_size = 20  # Choose how many combinations to run
# random.seed(42)
# sampled_combinations = random.sample(all_combinations, sample_size)
# for weights in sampled_combinations:
#     w1, w2, w3, w4, w5, w6, w7, w8 = weights
#     Weights = [w1, w2, w3, w4, w5, w6, w7, w8]
#     instance_id += 1

# NumberStMax = 5
# results = []
# instance_id = 0
# w5, w6, w7, w8 = 0.25, 0.25, 0.25, 0.25
#
# for w1 in [0.25, 10, 100]:
#     for w2 in [0.01, 0.25, 10, 100, 10000]:
#         for w3 in [0.01, 0.25, 1, 10, 100]:
#             for w4 in [0.00025, 0.25]:
#                 Weights = [w1, w2, w3, w4, w5, w6, w7, w8]
#                 instance_id += 1
#
#                 try:
#                     # Build and solve model
#                     model_1, x_s, y_os, z_sor, h_sov = build_model(
#                         Stations, OilSpills, Resources, Vehicles, Weights,
#                         v_o_n, eta_o, t_os_n, gamma, M, demand_or, demand_ov, L_p_or,
#                         NumberStMax, A_sr, nH, nUN, nQ, Q_vr, n_vs,
#                         F_s, C_r, Eff_sor, pn_sor, c_v, Distance, DistanceMax,
#                         'model_p')
#
#                     model_objectives, coverage_pct, resource_stockpile_r, x_s1, y_os1, z_sor_lamoscad, h_sov_lamoscad, solution_values = solve_model(
#                         model_1, x_s, y_os, z_sor, h_sov, OilSpills
#                     )
#
#                     mean_rt = utility_functions.compute_mean_response_time(y_os1, spill_df, station_df)
#
#                     results.append({
#                         "Instance": instance_id,
#                         "w1": w1, "w2": w2, "w3": w3, "w4": w4,
#                         "w5": w5, "w6": w6, "w7": w7, "w8": w8,
#                         "Objective_1": model_objectives[0],
#                         "Objective_2": model_objectives[1],
#                         "Coverage_%": coverage_pct,
#                         "Mean_Response_Time": mean_rt
#                     })
#
#                 except Exception as e:
#                     print(f"Failed for instance {instance_id} with weights {Weights}: {e}")
#                     results.append({
#                         "Instance": instance_id,
#                         "w1": w1, "w2": w2, "w3": w3, "w4": w4,
#                         "w5": w5, "w6": w6, "w7": w7, "w8": w8,
#                         "Objective_1": None,
#                         "Objective_2": None,
#                         "Coverage_%": None,
#                         "Mean_Response_Time": None,
#                         "Error": str(e)
#                     })
#
# df_sensitivity_of_weight_fig8 = pd.DataFrame(results)
# df_sensitivity_of_weight_fig8.to_excel('../results/artifacts/df_weight_sensitivity_150_fig8.xlsx')
df_fig8 = pd.read_excel('../results/artifacts/df_weight_sensitivity_150_fig8.xlsx')
draw_parallel_coordinate_plot(df_fig8)

# #%% ----------------------------------------------- Figure 9 ---------------------------------------------------------
# print("Running optimization models ")
# for m1 in ['model_p']:  # , 'model_2', 'model_5', 'model_6']:
#     print(f'Running {m1}')
#     time_start_gurobi = time.time()
#     model_1, x_s, y_os, z_sor, h_sov = build_model(Stations, OilSpills, Resources, Vehicles, W,
#                                                    v_o_n, eta_o, t_os_n, gamma, M, demand_or, demand_ov, L_p_or,
#                                                    NumberStMax_dict[m1], A_sr, nH, nUN, nQ, Q_vr, n_vs,
#                                                    F_s, C_sr, Eff_sor, pn_sor, c_v, Distance, DistanceMax_dict[m1], m1)
#
#     model_objectives, coverage_percentage, resource_stockpile_r, x_s1, y_os1, z_sor1, h_sov1, solution_values \
#         = solve_model(model_1, x_s, y_os, z_sor, h_sov, OilSpills)
#     # Draw the network
#     mean_response_time = draw_network.draw_network_diagram(y_os1, spill_df, station_df, name=m1)
#     if m1 == 'model_p':
#         results_table5_lamoscad = preprocess_utils.convert_table5(
#             coverage_percentage, mean_response_time, 25, time.time()-time_start_gurobi, len(x_s1), 'lamoscad')
#         h_sov_lamoscad = h_sov1
#         z_sor_lamoscad = z_sor1
#         print('z_sor_lamoscad: ', z_sor_lamoscad)
#
#         # run another run of model_2 (for Fig 9a)
#         original_w2 = W[1]
#         print(f'original_w2 is {original_w2} in {m1}')
#         print('W before assign new value in w2', W)
#         W[1] = w2 = 0.02
#         print('W after assign', W)
#         model_1, x_s, y_os, z_sor, h_sov = build_model(Stations, OilSpills, Resources, Vehicles, W,
#                                                        v_o_n, eta_o, t_os_n, gamma, M, demand_or, demand_ov, L_p_or,
#                                                        NumberStMax_dict[m1], A_sr, nH, nUN, nQ, Q_vr, n_vs,
#                                                        F_s, C_sr, Eff_sor, pn_sor, c_v, Distance, DistanceMax_dict[m1],
#                                                        m1)
#
#         model_objectives, coverage_percentage, resource_stockpile_r, x_s1, y_os1, z_sor1, h_sov1, solution_values \
#             = solve_model(model_1, x_s, y_os, z_sor, h_sov, OilSpills)
#         mean_response_time = draw_network_diagram(y_os1, spill_df, station_df, name='fig9a')
#         W[1] = original_w2
#
#     Objective1_dict[m1] = model_objectives[0]
#     Objective2_dict[m1] = model_objectives[1]
#     Coverage_dict[m1] = coverage_percentage
#     Mean_response_time_dict[m1] = mean_response_time


#%% ----------------------------------------------- Section 4.3.3. -----------------------------------------------------
#%% ----------------------------------------------- Figure 10. Model Performance boxplot -------------------------------
"""
# create 10 different dataset
df_pareto_front_fig11 = pd.DataFrame(index=range(10),
                                     columns=["Coverage%_c", "Coverage%_p", "Obj2_c", "Obj2_p", "Mean_RT_c", "Mean_RT_p"])
# old one  W = [10 * 0.25, 10 * 0.25, 0.25, 0.25 * 10 ** -2, 0.25, 0.25, 0.25, 0.25]
W = [2.5, .25, 1, 0.0000025, 2500, 1, 1, 100]

for SetName in ['setA', 'setB', 'setC', 'setD', 'setE', 'setF', 'setG', 'setH', 'setI', 'setJ']:
    for stationType in ['current', 'proposed']:  #
        # try:
        with open(f"../data/10 Sets of spills/preprocessed_data_{SetName}_{stationType}.pkl", "rb") as f:
            data = pickle.load(f)
        print(f'\n\nSolving {SetName} for \'{stationType}\' facility setup')
        OilSpills, Resources, demand_or, demand_ov = data["OilSpills"], data["Resources"], data["demand_or"], data[
            "demand_ov"]
        coordinates_spill, Sensitivity_R = data["coordinates_spill"], data["Sensitivity_R"]
        v_o, v_o_n, eta_o = data["v_o"], data["v_o_n"], data["eta_o"]
        spill_df = preprocess_utils.create_spill_dataframe(coordinates_spill, v_o, Sensitivity_R)

        cfg = config_loader.load_config("../src/config/model_config.yaml")
        W, M, gamma = cfg["general"]["weights"], cfg["general"]["M"], cfg["general"]["gamma"]
        DistanceMax = cfg["general"]["DistanceMax"] if stationType == 'proposed' else 15  # 15 for current facility setup
        NumberStMax = 5 if stationType == 'proposed' else 4
        nH, nUN, nQ = cfg["general"]["nH"], cfg["general"]["nUN"], cfg["general"]["nQ"]
        Vehicles, c_v, C_r = cfg["assets"]["vehicles"], cfg["assets"]["c_v"], cfg["assets"]["C_r"]
        Q_vr = {(v, r): cfg["assets"]["Q_vr"][v][r] for v in cfg["assets"]["Q_vr"] for r in cfg["assets"]["Q_vr"][v]}

        # if stationType == 'current':
        #     with open(f"../data/preprocessed_data_model_c.pkl", "rb") as f:
        #         data1 = pickle.load(f)
        #     (Stations, coordinates_st, A_sr, Eff_sor,
        #      Distance, Distance_n,
        #      t_os, F_s, t_os_n, pn_sor) = (data1["Stations"], data1["coordinates_st"], data1["A_sr"], data1["Eff_sor"],
        #                                    data1["Distance"], data1["Distance_n"], data1["t_os"], data1["F_s"],
        #                                    data1["t_os_n"], data1["pn_sor"])
        #     station_df = preprocess_utils.create_station_dataframe(coordinates_st)
        #     n_vs = {(v, s): cfg["assets"]["n_vs_values"][v] for v in Vehicles for s in Stations}
        #     L_p_or = {(o, r): cfg["L_p_or_values"]["c_i"] for o in OilSpills for r in ["c", "i"]}
        #     L_p_or.update({(o, r): float("inf") for o in OilSpills for r in ["m"]})
        #
        # else:
        Stations, coordinates_st, A_sr, Eff_sor = data["Stations"], data["coordinates_st"], data["A_sr"], data[
            "Eff_sor"]
        Distance, Distance_n, t_os, F_s, t_os_n, pn_sor = data["Distance"], data["Distance_n"], data["t_os"], data[
                                                            "F_s"], data["t_os_n"], data["pn_sor"]
        station_df = preprocess_utils.create_station_dataframe(coordinates_st)
        n_vs = {(v, s): cfg["assets"]["n_vs_values"][v] for v in Vehicles for s in Stations}
        L_p_or = {(o, r): cfg["L_p_or_values"]["c_i"] for o in OilSpills for r in ["c", "i"]}
        L_p_or.update({(o, r): float("inf") for o in OilSpills for r in ["m"]})

        model_1, x_s, y_os, z_sor, h_sov = build_model(Stations, OilSpills, Resources, Vehicles, W,
                                                       v_o_n, eta_o, t_os_n, gamma, M, demand_or, demand_ov, L_p_or,
                                                       NumberStMax, A_sr, nH, nUN, nQ, Q_vr, n_vs,
                                                       F_s, C_r, Eff_sor, pn_sor, c_v, Distance, DistanceMax,
                                                       'model_p' if stationType == 'proposed' else 'model_c')

        model_objectives, coverage_percentage, resource_stockpile_r, x_s1, y_os1, z_sor_lamoscad, h_sov_lamoscad, solution_values \
            = solve_model(model_1, x_s, y_os, z_sor, h_sov, OilSpills, needMultiSolutions=True, uncertaintyEvaluation=True)
        mean_response_time = draw_network_diagram(y_os1, spill_df, station_df, name='model_p')
        obj2_tactical_cost = int(solution_values[0][1] - W[3] * sum(F_s[s] * x_s[s].X for s in Stations))
        set_mapping = {'setA': 0, 'setB': 1, 'setC': 2, 'setD': 3, 'setE': 4,
                       'setF': 5, 'setG': 6, 'setH': 7, 'setI': 8, 'setJ': 9}
        i = set_mapping.get(SetName, -1)

        col_suffix = "_p" if stationType == "proposed" else "_c"
        df_pareto_front_fig11.iloc[
            i, df_pareto_front_fig11.columns.get_loc(f"Coverage%{col_suffix}")] = coverage_percentage
        df_pareto_front_fig11.iloc[i, df_pareto_front_fig11.columns.get_loc(f"Obj2{col_suffix}")] = obj2_tactical_cost
        df_pareto_front_fig11.iloc[i, df_pareto_front_fig11.columns.get_loc(f"Mean_RT{col_suffix}")] = mean_response_time
        print('coverage_percentage, obj2_tactical_cost, mean_response_time: ',
              coverage_percentage, obj2_tactical_cost, mean_response_time)

        # except KeyError as e:
        #     print(e)

df_pareto_front_fig11.to_excel("../results/artifacts/df_pareto_front_fig11_proposed_vs_current.xlsx",
                               index_label="SpillSet")
"""
df_pareto_front_fig11 = pd.read_excel("../results/artifacts/df_pareto_front_fig11_proposed_vs_current.xlsx")
draw_boxplot(df_pareto_front_fig11)
# data_compare = pd.read_excel('../data/processed/Boxplot_data.xlsx', sheet_name='Sheet1', skiprows=[0])

#%% ------------------------------- Table 9. hierarchical optimization -------------------------------------------------
# tab9_hierarchical_optimization = pd.DataFrame()
#
#%% ------------------------------- Figure 11. Pareto Frontier ---------------------------------------------------------
# max_coverage_lists = df_pareto_front_fig11['Max coverage'].tolist()
# min_cost_lists = df_pareto_front_fig11['Min cost'].tolist()
# flat_list_max_coverage = [item for sublist in max_coverage_lists for item in sublist]
# flat_list_min_cost = [item for sublist in min_cost_lists for item in sublist]
#
# # Creating excel file
# with pd.ExcelWriter('../results/4.3_sensitivity_analysis.xlsx', engine='openpyxl', mode='w') as writer:
#     tab8_num_facilities.to_excel(writer, sheet_name='tab8. sensitivity num facility', index=False)
#     tab9_hierarchical_optimization.to_excel(writer, sheet_name='tab9. hierarchical opt', index=False)

