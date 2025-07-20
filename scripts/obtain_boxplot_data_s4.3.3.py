""" Obtain boxplot data for Section 4.3.3"""

#%%
from src.models.model_lamoscad import build_model, solve_model
from src.utils.utility_functions import compute_mean_response_time
from src.preprocessing import data_loader, preprocess_utils
from src.visualization.draw_boxplot import draw_boxplot
from src.config import config_loader
import pickle
import pandas as pd

#%%
# create 10 different dataset
df_boxplot_fig10 = pd.DataFrame(index=range(10),
                                     columns=["Coverage%_c", "Coverage%_p", "Obj2_c", "Obj2_p", "Mean_RT_c", "Mean_RT_p"])
# df_pareto_front_fig11 = pd.DataFrame(index=range(10), columns=["SetName", "Max Coverage", "Min Cost"])
all_solution_rows = []


for SetName in ['setA', 'setB', 'setC', 'setD', 'setE', 'setF', 'setG', 'setH', 'setI', 'setJ']:
    for stationType in ['proposed', 'current']:  #
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
        M, gamma = cfg["general"]["M"], cfg["general"]["gamma"]
        DistanceMax = cfg["general"]["DistanceMax"]  # if stationType == 'proposed' else 15
        NumberStMax = 5 if stationType == 'proposed' else 4  # cfg["general"]["NumberStMax"]   #
        W = [25, 2.5, .25, 0.0025,	0.5,	0.25,	0.20,	0.25] if stationType == 'proposed' else [2.5, 2.5, 2.5, 0.0025, 0.5, 100, 0.5, 0.25]
        #
        nH, nUN, nQ = cfg["general"]["nH"], cfg["general"]["nUN"], cfg["general"]["nQ"]
        Vehicles, c_v, C_r = cfg["assets"]["vehicles"], cfg["assets"]["c_v"], cfg["assets"]["C_r"]
        Q_vr = {(v, r): cfg["assets"]["Q_vr"][v][r] for v in cfg["assets"]["Q_vr"] for r in cfg["assets"]["Q_vr"][v]}

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

        mean_response_time = compute_mean_response_time(y_os1, spill_df, station_df)
        obj2_tactical_cost = int(solution_values[0][1] - W[3] * sum(F_s[s] * x_s[s].X for s in Stations))
        set_mapping = {'setA': 0, 'setB': 1, 'setC': 2, 'setD': 3, 'setE': 4,
                       'setF': 5, 'setG': 6, 'setH': 7, 'setI': 8, 'setJ': 9}
        i = set_mapping.get(SetName, -1)

        col_suffix = "_p" if stationType == "proposed" else "_c"
        df_boxplot_fig10.iloc[
            i, df_boxplot_fig10.columns.get_loc(f"Coverage%{col_suffix}")] = coverage_percentage
        df_boxplot_fig10.iloc[i, df_boxplot_fig10.columns.get_loc(f"Obj2{col_suffix}")] = obj2_tactical_cost
        df_boxplot_fig10.iloc[i, df_boxplot_fig10.columns.get_loc(f"Mean_RT{col_suffix}")] = mean_response_time

        print('coverage_percentage, obj2_tactical_cost, mean_response_time: ',
              coverage_percentage, obj2_tactical_cost, mean_response_time)

        # except KeyError as e:
        #     print(e)

draw_boxplot(df_boxplot_fig10)
df_boxplot_fig10.to_excel("../results/artifacts/df_boxplot_fig10.xlsx", index_label="SpillSet")
