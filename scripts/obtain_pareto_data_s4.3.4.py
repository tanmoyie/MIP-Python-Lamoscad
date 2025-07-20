""" Obtain Data of Pareto Front """
from src.models.model_lamoscad import build_model, solve_model
from src.preprocessing import data_loader, preprocess_utils
from src.visualization.plot_pareto_frontier import plot_pareto_frontier
from src.config import config_loader
import pickle
import pandas as pd

#%%
# create 10 different dataset
df_pareto_front_fig11 = pd.DataFrame(index=range(10), columns=["SetName", "Max Coverage", "Min Cost"])
all_solution_rows = []
W = [0.1,	2.5,	2.5,	0.000000025,	0.5, 100,	0.5,	0.25]

for SetName in ['setA', 'setB', 'setC', 'setD', 'setE', 'setF', 'setG', 'setH', 'setI', 'setJ']:
    stationType = 'proposed'
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
    DistanceMax = cfg["general"]["DistanceMax"]
    NumberStMax = 5
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
                                                   'model_p')

    _, _, _, _, _, _, _, solution_values \
        = solve_model(model_1, x_s, y_os, z_sor, h_sov, OilSpills, needMultiSolutions=True, uncertaintyEvaluation=True)
    print('solution_values: ', solution_values)

    seen_pairs = set()
    unique_count = 0
    for k, sol in enumerate(solution_values):
        max_coverage = sol[0]
        min_cost = sol[1]
        pair = (max_coverage, min_cost)

        if pair not in seen_pairs:
            seen_pairs.add(pair)
            all_solution_rows.append({
                "SetName": f"{SetName} {k + 1}",
                "Max Coverage": max_coverage,
                "Min Cost": min_cost
            })
            unique_count += 1
            if unique_count >= 3:
                break
    # except KeyError as e:
    #     print(e)

df_pareto_front_fig11 = pd.DataFrame(all_solution_rows)
plot_pareto_frontier(df_pareto_front_fig11)
df_pareto_front_fig11.to_excel("../results/artifacts/df_pareto_front_fig11.xlsx", index_label="SpillSet")
