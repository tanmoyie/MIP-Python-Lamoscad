""" Obtain data needed for Fig8 """
import pandas as pd
from src.config import config_loader
from src.utils import utility_functions
import pickle
from src.visualization.draw_parallel_coordinate_plot import draw_parallel_coordinate_plot
from src.preprocessing import data_loader, preprocess_utils
from src.models.model_lamoscad import build_model, solve_model
#%% Data loading
oil = 100

sta = 20
# Load data for this oil size
with open(f"../data/large scale processed dataset/preprocessed_data_o{oil}_s{sta}.pkl", "rb") as f:
    data = pickle.load(f)

OilSpills, Resources, demand_or, demand_ov = data["OilSpills"], data["Resources"], data["demand_or"], data["demand_ov"]
coordinates_spill, Sensitivity_R = data["coordinates_spill"], data["Sensitivity_R"]
v_o, v_o_n, eta_o = data["v_o"], data["v_o_n"], data["eta_o"]
Stations, coordinates_st, A_sr, Eff_sor = data["Stations"], data["coordinates_st"], data["A_sr"], data["Eff_sor"]
Distance, Distance_n, t_os, F_s, t_os_n, pn_sor = data["Distance"], data["Distance_n"], data["t_os"], data["F_s"], data["t_os_n"], data["pn_sor"]

cfg = config_loader.load_config("../src/config/model_config.yaml")
W, M, gamma = cfg["general"]["weights"], cfg["general"]["M"], cfg["general"]["gamma"]
DistanceMax = cfg["general"]["DistanceMax"]
nH, nUN, nQ = cfg["general"]["nH"], cfg["general"]["nUN"], cfg["general"]["nQ"]
Vehicles, c_v, C_r = cfg["assets"]["vehicles"], cfg["assets"]["c_v"], cfg["assets"]["C_r"]
Q_vr = {(v, r): cfg["assets"]["Q_vr"][v][r] for v in cfg["assets"]["Q_vr"] for r in cfg["assets"]["Q_vr"][v]}
n_vs = {(v, s): cfg["assets"]["n_vs_values"][v] for v in Vehicles for s in Stations}
L_p_or = {(o, r): cfg["L_p_or_values"]["c_i"] for o in OilSpills for r in ["c", "i"]}
L_p_or.update({(o, r): float("inf") for o in OilSpills for r in ["m"]})

spill_df = preprocess_utils.create_spill_dataframe(coordinates_spill, v_o, Sensitivity_R)
station_df = preprocess_utils.create_station_dataframe(coordinates_st)
NumberStMax = 5
results = []

#%% ------------------------------ Weight combinations (if I wanna generate lots of weight combinations ----------------
# instance_id = 0
# w1_list = [.0025, 0.1, 0.2, .25, 0.5, 0.8, 1, 2.5, 100]
# w2_list = [.0025, 0.1, 0.2, .25, 0.5, 0.8, 1, 2.5, 100]
# w3_list = [.0025, 0.1, 0.2, .25, 0.5, 0.8, 1, 2.5, 100]
# w4_list = [0.25e-7, .0025, 0.01, 0.09, 0.1, 0.2, 1]
# w5_list = [0.0001, 0.001, 0.25, 0.5, 0.75, 1]
# w6_list = [0.0025, 0.25, 0.5, 0.75, 1, 100]
# w7_list = [0.0025, 0.025, 0.25, 0.5, 0.75, 1]
# w8_list = [0.001, 0.25, 0.5, 0.75, 1, 100]
# all_combinations = list(itertools.product(w1_list, w2_list, w3_list, w4_list,
#     w5_list, w6_list, w7_list, w8_list))
# sample_size = 20  # Choose how many combinations to run ++
# random.seed(42)
# sampled_combinations = random.sample(all_combinations, sample_size)
# for weights in sampled_combinations:
#     w1, w2, w3, w4, w5, w6, w7, w8 = weights
#     Weights = [w1, w2, w3, w4, w5, w6, w7, w8]
#     instance_id += 1

# ---------------------------------- or Weight combinations (a total of 150) used in the manuscript --------------------
instance_id = 0
w5, w6, w7, w8 = 0.25, 0.25, 0.25, 0.25
for w1 in [0.25, 10, 100]:
    for w2 in [0.01, 0.25, 10, 100, 10000]:
        for w3 in [0.01, 0.25, 1, 10, 100]:
            for w4 in [0.00025, 0.25]:
                Weights = [w1, w2, w3, w4, w5, w6, w7, w8]
                print('instance_id', instance_id)
                instance_id += 1
                try:
                    # Build and solve model
                    model_1, x_s, y_os, z_sor, h_sov = build_model(
                        Stations, OilSpills, Resources, Vehicles, Weights,
                        v_o_n, eta_o, t_os_n, gamma, M, demand_or, demand_ov, L_p_or,
                        NumberStMax, A_sr, nH, nUN, nQ, Q_vr, n_vs,
                        F_s, C_r, Eff_sor, pn_sor, c_v, Distance, DistanceMax,
                        'model_p')

                    model_objectives, coverage_pct, resource_stockpile_r, x_s1, y_os1, z_sor_lamoscad, h_sov_lamoscad, solution_values = (
                        solve_model(model_1, x_s, y_os, z_sor, h_sov, OilSpills))

                    mean_rt = utility_functions.compute_mean_response_time(y_os1, spill_df, station_df)

                    results.append({
                        "Instance": instance_id,
                        "w1": w1, "w2": w2, "w3": w3, "w4": w4,
                        "w5": w5, "w6": w6, "w7": w7, "w8": w8,
                        "Objective_1": model_objectives[0],
                        "Objective_2": model_objectives[1],
                        "Coverage_%": coverage_pct,
                        "Mean_Response_Time": mean_rt})

                except Exception as e:
                    print(f"Failed for instance {instance_id} with weights {Weights}: {e}")
                    results.append({"Instance": instance_id,
                        "w1": w1, "w2": w2, "w3": w3, "w4": w4,
                        "w5": w5, "w6": w6, "w7": w7, "w8": w8,
                        "Objective_1": None,
                        "Objective_2": None,
                        "Coverage_%": None,
                        "Mean_Response_Time": None,
                        "Error": str(e)})

df_sensitivity_of_weight_fig8 = pd.DataFrame(results)
draw_parallel_coordinate_plot(df_sensitivity_of_weight_fig8)
df_sensitivity_of_weight_fig8.to_excel(f'../results/artifacts/df_weight_sensitivity_{instance_id}_fig8.xlsx')
