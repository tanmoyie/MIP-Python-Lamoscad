"""
Section 4.3

"""
# %% 📦 Imports & Configurations
from src.models.model_lamoscad import build_model, solve_model
from src.models.model_mclp import build_model_mclp, solve_model_mclp
from src.preprocessing import data_loader, preprocess_utils
from src.visualization import draw_network, plot_pareto_frontier, plot_parallel_coordinate_plot
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

#%% -------------------------------------------- Section 4.3.1 Impact of number of facilities --------------------------
# Data
NumberStMaxL = [15, 12, 8, 5, 4, 2]
number_of_spillsL = [100, 100, 100, 100, 100, 100]
select_1sL =[]
coverage_percentageL =[]
MeanResponseTimeL =[]

# 200 spills
number_of_spillsL = [200, 200, 200, 200, 200, 200]
DistanceMax = 10

BigM = 10**20
current_vs_proposed = 'proposed' # current
MaxFO = 1
Budget = 10**12
W = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
QuantityMin = 2

for i in range(len(NumberStMaxL)):
    model01, select, deploy, mvars, names, values, objValues, \
            spill_df, station_df, cover_1s, select_1s, deploy_1s, MeanResponseTime, coverage_percentage, assignment  \
        = Model.solve(Stations, OilSpills, ResourcesD, coordinates_st, coordinates_spill, SizeSpill, SizeSpill_n,
              Demand, Sensitivity_R, Sensitivity_n, Eff, Effectiveness_n, Availability, NumberStMaxL[i], Distance, Distance_n,
              W, QuantityMin, DistanceMax, Cf_s, CostU, Budget,
              BigM, MaxFO)
    select_1sL.append(select_1s.index)
    coverage_percentageL.append(coverage_percentage)
    MeanResponseTimeL.append(MeanResponseTime)

table8_df_sensitivity_num_facilities = pd.DataFrame(index=range(len(NumberStMaxL)),
                                              columns=['Number of facilities', 'Number of spills','Open facility', 'Coverage (%)', 'Mean Response time (in hours)', 'Cost investment (in million CAD)'])
for i in range(6):
    table8_df_sensitivity_num_facilities.iloc[i,0] = NumberStMaxL[i]
    table8_df_sensitivity_num_facilities.iloc[i,1] = number_of_spillsL[i]
    table8_df_sensitivity_num_facilities.iloc[i,2] = select_1sL[i]
    table8_df_sensitivity_num_facilities.iloc[i,3] = coverage_percentageL[i]
    table8_df_sensitivity_num_facilities.iloc[i,4] = MeanResponseTimeL[i]
    table8_df_sensitivity_num_facilities.iloc[i,5] = 0

# %% ------------------------------------------ Section 4.3.2 Weight and sensitive oil spills  -------------------------
# Data
# Figure 8
plot_parallel_coordinate_plot.plot_parallel_coordinate_plot(data_co_sorted, data_MeanRT_sorted, data_cost_sorted)

#%% Figure 9
print("Running optimization models ")
for m1 in ['model_p']:  # , 'model_2', 'model_5', 'model_6']:
    print(f'Running {m1}')
    time_start_gurobi = time.time()
    model_1, x_s, y_os, z_sor, h_sov = build_model(Stations, OilSpills, Resources, Vehicles, W,
                                                   v_o_n, eta_o, t_os_n, gamma, M, demand_or, demand_ov, L_p_or,
                                                   NumberStMax_dict[m1], A_sr, nH, nUN, nQ, Q_vr, n_vs,
                                                   F_s, C_sr, Eff_sor, pn_sor, c_v, Distance, DistanceMax_dict[m1], m1)

    model_objectives, coverage_percentage, resource_stockpile_r, x_s1, y_os1, z_sor1, h_sov1, solution_values \
        = solve_model(model_1, x_s, y_os, z_sor, h_sov, OilSpills)
    # Draw the network
    mean_response_time = draw_network.draw_network_diagram(y_os1, spill_df, station_df, name=m1)
    if m1 == 'model_p':
        results_table5_lamoscad = preprocess_utils.convert_table5(
            coverage_percentage, mean_response_time, 25, time.time()-time_start_gurobi, len(x_s1), 'lamoscad')
        h_sov_lamoscad = h_sov1
        z_sor_lamoscad = z_sor1
        print('z_sor_lamoscad: ', z_sor_lamoscad)

        # run another run of model_2 (for Fig 9a)
        original_w2 = W[1]
        print(f'original_w2 is {original_w2} in {m1}')
        print('W before assign new value in w2', W)
        W[1] = w2 = 0.02
        print('W after assign', W)
        model_1, x_s, y_os, z_sor, h_sov = build_model(Stations, OilSpills, Resources, Vehicles, W,
                                                       v_o_n, eta_o, t_os_n, gamma, M, demand_or, demand_ov, L_p_or,
                                                       NumberStMax_dict[m1], A_sr, nH, nUN, nQ, Q_vr, n_vs,
                                                       F_s, C_sr, Eff_sor, pn_sor, c_v, Distance, DistanceMax_dict[m1],
                                                       m1)

        model_objectives, coverage_percentage, resource_stockpile_r, x_s1, y_os1, z_sor1, h_sov1, solution_values \
            = solve_model(model_1, x_s, y_os, z_sor, h_sov, OilSpills)
        mean_response_time = draw_network_diagram(y_os1, spill_df, station_df, name='fig9a')
        W[1] = original_w2

    Objective1_dict[m1] = model_objectives[0]
    Objective2_dict[m1] = model_objectives[1]
    Coverage_dict[m1] = coverage_percentage
    Mean_response_time_dict[m1] = mean_response_time


#%% ------------------------------------------ Section 4.3.3. --------------------------------------------
#%% Figure 10
# create 10 different dataset
pareto_front_df = pd.DataFrame(index=range(10), columns=['Max coverage', 'Min cost'])
W = [10 * 0.25, 10 * 0.25, 0.25, 0.25 * 10 ** -2, 0.25, 0.25, 0.25, 0.25]

for SetName in ['setA', 'setB', 'setC', 'setD', 'setE', 'setF', 'setG', 'setH', 'setI', 'setJ']:

    # need to regenerate oil spill related data for new oil spills
    # read csv file for newly created 10 sets
    spill_data = pd.read_csv(f'data/10 Sets of spills/{SetName}_data_100_oil_spills.csv')
    coordinates_spill = custom_func.extract_spill_coordinate(spill_data)
    Stations, OilSpills, Resources, demand_or, demand_ov, A_sr, Eff_sor, Distance, t_os, F_s, C_sr \
        = custom_func.generate_input_data(station_data, spill_data, input_parameters)
    Sensitivity_R = custom_func.calculate_sensitivity(list(coordinates_spill.values()), sensitivity_dataR)
    v_o_n = dict(zip(OilSpills, custom_func.normalize(SizeSpill_R, min(SizeSpill_R), max(SizeSpill_R))))  # SizeSpill_n
    eta_o = dict(zip(OilSpills, custom_func.normalize(Sensitivity_R, min(Sensitivity_R), max(Sensitivity_R))))
    t_os_n = dict(zip(t_os.keys(), custom_func.normalize(t_os.values(), min(t_os.values()), max(t_os.values()))))
    pn_sor = custom_func.compute_penalty(Stations, OilSpills, Resources)

    model_1, x_s, y_os, z_sor, h_sov = build_model(Stations, OilSpills, Resources, Vehicles, W,
                                                   v_o_n, eta_o, t_os_n, gamma, M, demand_or, demand_ov, L_p_or,
                                                   NumberStMax_dict[m1], A_sr, nH, nUN, nQ, Q_vr, n_vs,
                                                   F_s, C_sr, Eff_sor, pn_sor, c_v, Distance, DistanceMax_dict[m1], m1)

    model_objectives, coverage_percentage, resource_stockpile_r, x_s1, y_os1, z_sor1, h_sov1, solution_values \
        = solve_model(model_1, x_s, y_os, z_sor, h_sov, OilSpills)

    set_mapping = {'setA': 0, 'setB': 1, 'setC': 2, 'setD': 3, 'setE': 4,
                   'setF': 5, 'setG': 6, 'setH': 7, 'setI': 8, 'setJ': 9}
    i = set_mapping.get(SetName, -1)

    pareto_front_df.iloc[i, 0] = [j[0] for j in solution_values]
    pareto_front_df.iloc[i, 1] = [j[1] for j in solution_values]


#%% Figure 11
print(pareto_front_df)
pareto_front_df.to_excel('results/pareto_front_df_fig11.xlsx')
df = pd.read_excel('results/pareto_front_df_fig11.xlsx')
max_coverage_lists = pareto_front_df['Max coverage'].tolist()
min_cost_lists = pareto_front_df['Min cost'].tolist()
flat_list_max_coverage = [item for sublist in max_coverage_lists for item in sublist]
flat_list_min_cost = [item for sublist in min_cost_lists for item in sublist]

#%% Table 9
table9_df_hierarchical_optimization = pd.DataFrame()

# Creating excel file
table8_df_sensitivity_num_facilities
table9_df_hierarchical_optimization
