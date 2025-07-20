"""
Section 4.3.
Filename= perform_sensitivity_analysis_s4.3
This script will produce all tables and figures in Section 4.3
"""
from src.models.model_insensitive import build_model_insensi, solve_model_insensi
from src.models.model_lamoscad_large_scale import build_model, solve_model
from src.preprocessing import data_loader, preprocess_utils
from src.visualization.draw_network import draw_network_diagram
from src.visualization.draw_parallel_coordinate_plot import draw_parallel_coordinate_plot
from src.visualization.draw_boxplot import draw_boxplot
from src.visualization.plot_pareto_frontier import plot_pareto_frontier
from src import config
from src.config import config_loader
from src.utils import utility_functions
import pickle
import pandas as pd

#%% -------------------------------------------- Section 4.3.1 Impact of number of facilities --------------------------
# --------------------------------------------------- Table 8 ---------------------------------------------------------
results = []
for oil in [100, 200]:
    sta = 20
    print(f'Running on len(OilSpill)={oil}, N={sta}')
    # Load data for this oil size
    with open(f"../data/large scale processed dataset/preprocessed_data_o{oil}_s{sta}.pkl", "rb") as f:
        data = pickle.load(f)

    OilSpills, Resources, demand_or, demand_ov = data["OilSpills"], data["Resources"], data["demand_or"], data["demand_ov"]
    coordinates_spill, Sensitivity_R = data["coordinates_spill"], data["Sensitivity_R"]
    v_o, v_o_n, eta_o = data["v_o"], data["v_o_n"], data["eta_o"]
    Stations, coordinates_st, A_sr, Eff_sor = data["Stations"], data["coordinates_st"], data["A_sr"], data["Eff_sor"]
    Distance, Distance_n, t_os, F_s, t_os_n, pn_sor = data["Distance"], data["Distance_n"], data["t_os"], data["F_s"], data["t_os_n"], data["pn_sor"]

    cfg = config_loader.load_config("../src/config/model_config_4.3.yaml")
    W, M, gamma = [2.5, 2.5, 0.25, 0.09, 0.025, 25, 0.25, 0.25], cfg["general"]["M"], cfg["general"]["gamma"]
    DistanceMax = cfg["general"]["DistanceMax"]
    nH, nUN, nQ = cfg["general"]["nH"], cfg["general"]["nUN"], cfg["general"]["nQ"]
    Vehicles, c_v, C_r = cfg["assets"]["vehicles"], cfg["assets"]["c_v"], cfg["assets"]["C_r"]
    Q_vr = {(v, r): cfg["assets"]["Q_vr"][v][r] for v in cfg["assets"]["Q_vr"] for r in cfg["assets"]["Q_vr"][v]}
    n_vs = {(v, s): cfg["assets"]["n_vs_values"][v] for v in Vehicles for s in Stations}
    L_p_or = {(o, r): cfg["L_p_or_values"]["c_i"] for o in OilSpills for r in ["c", "i"]}
    L_p_or.update({(o, r): float("inf") for o in OilSpills for r in ["m"]})

    spill_df = preprocess_utils.create_spill_dataframe(coordinates_spill, v_o, Sensitivity_R)
    station_df = preprocess_utils.create_station_dataframe(coordinates_st)

    for NumberStMax in [15, 12, 8, 5, 4, 2]:
        model_1, x_s, y_os, z_sor, h_sov = build_model(
            Stations, OilSpills, Resources, Vehicles, W,
            v_o_n, eta_o, t_os_n, gamma, M, demand_or, demand_ov, L_p_or,
            NumberStMax, A_sr, nH, nUN, nQ, Q_vr, n_vs,
            F_s, C_r, Eff_sor, pn_sor, c_v, Distance, DistanceMax, 'model_p')

        _, coverage_pct, _, x_s1, y_os1, _, _, _, _ = solve_model(
            model_1, x_s, y_os, z_sor, h_sov, OilSpills)
        mean_response_time = utility_functions.compute_mean_response_time(y_os1, spill_df, station_df)

        results.append({
            "Max_Facilities": NumberStMax,
            "Num_Spills": len(OilSpills),
            "Open_Facilities": list(x_s1.index),
            "Coverage_%": coverage_pct,
            "Mean_Response_Time": mean_response_time,
        })

tab8_num_facilities_sensitivity = pd.DataFrame(results)

# %% ------------------------------------------ Section 4.3.2 Weight and sensitive oil spills  -------------------------
# --------------------------------------------------- Figure 8. PCP -------------------------------------------
df_fig8 = pd.read_excel('../results/artifacts/df_weight_sensitivity_150_fig8.xlsx')
draw_parallel_coordinate_plot(df_fig8)

# #%% ----------------------------------------------- Figure 9. ND --------------------------------------------
name, W[0], W[1], W[2], DistanceMax = 'fig9a', 2.5, -2, 0.25, 20  # , 0.0025, 0 , W[0], W[1]
model_1, x_s, y_os, z_sor, h_sov = build_model_insensi(Stations, OilSpills, Resources, Vehicles, W,
                                               v_o_n, eta_o, t_os_n, gamma, M, demand_or, demand_ov, L_p_or,
                                               5, A_sr, nH, nUN, nQ, Q_vr, n_vs,
                                               F_s, C_r, Eff_sor, pn_sor, c_v, Distance, DistanceMax, 'model_p',
                                               insensitive=True)
model_objectives, coverage_percentage, resource_stockpile_r, x_s1, y_os1, z_sor1, h_sov1, solution_values \
    = solve_model_insensi(model_1, x_s, y_os, z_sor, h_sov, OilSpills)
_ = draw_network_diagram(y_os1, spill_df, station_df, name=name)
# fig9b is same as Fig5d (proposed model)

#%% ----------------------------------------------- Section 4.3.3. -----------------------------------------------------
# ----------------------------------------------- Figure 10. Model Performance boxplot -------------------------------
df_boxplot_fig10 = pd.read_excel("../results/artifacts/df_boxplot_fig10.xlsx")
draw_boxplot(df_boxplot_fig10)

#%% ------------------------------- Figure 11. Pareto Frontier ---------------------------------------------------------
df_pareto_front_fig11 = pd.read_excel("../results/artifacts/df_pareto_front_fig11.xlsx")
plot_pareto_frontier(df_pareto_front_fig11)

#%% Creating excel file
with pd.ExcelWriter('../results/sensitivity_analysis_s4.3.xlsx',
                    engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    tab8_num_facilities_sensitivity.to_excel(writer, sheet_name='tab8. sensitivity num facility', index=False)

print('Run completed on perform_sensitivity_analysis_s4.3')
