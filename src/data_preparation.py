""" File name: data_generation.py

Outline:
Calculating Demand, effectiveness, availability, distance, fixed and variable costs

Developer: Tanmoy Das
Date: March 2023

"""

import custom_func
import data_preparation
import custom_func
from model import Model
from model_w import ModelW
from draw_network import DrawNetworkDiagram

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class DataPreparation:
    def generate_input_data(station_data, spill_data, input_parameters):
        """
        This function will return all input parameters needed for the model.
        :param station_data:
        :param spill_data:
        :param input_parameters: Excel file containing data for calculating distance between spill and facility; demand,
        effectiveness and availability
        :return: Stations, OilSpills, ResourcesD, Demand, Availability, Eff, Distance, TimeR, Cf_s, Cu_sor
        """
        Stations = list(station_data['Station #'].unique())
        print(Stations)
        OilSpills = list(spill_data['Spill #'].unique())
        ResourcesD = ['m', 'c', 'i']

        #%% Demand[o,r]
        Demand_df = input_parameters[['Spill #', 'm', 'c', 'i']]
        D_stacked_df = Demand_df.set_index('Spill #').stack()

        o_r_comb = [(o, r) for o in OilSpills for r in ResourcesD]
        Demand = {}
        for ii in range(len(o_r_comb)):
            Demand[(o_r_comb[ii])] = D_stacked_df[ii]

        # %% Effectiveness[s,r] need to add o later ++
        Eff_df = input_parameters[['Station', 'm.1', 'c.1', 'i.1']]
        Eff_df.columns = ['Station', 'm', 'c', 'i']
        Eff_stacked_df = Eff_df.set_index('Station').stack()
        s_r_comb = [(s, r) for s in Stations for r in ResourcesD]
        Eff = {}
        for ii in range(len(s_r_comb)):
            Eff[(s_r_comb[ii])] = Eff_stacked_df[ii]
        # ++ Effectiveness could be more related to oil spill itself

        #%% Availability of resources in a station
        Ava_df = input_parameters[['Station.1', 'm.2', 'c.2', 'i.2']]
        Ava_df.columns = ['Station', 'm', 'c', 'i']
        Ava_stacked_df = Ava_df.set_index('Station').stack()
        Availability = {s_r_comb: Ava for ((s_r_comb), Ava) in zip(s_r_comb, Ava_stacked_df)}

        #%%
        # TimeWindowM = 6 * 60 if Tech_index == 'cdu' else (3 * 60 if Tech_index == 'cdu' else TimeRMax)
        # TimeR = {(st, os): custom_func.compute_distance(coordinates_st[st], coordinates_spill[os])
        #         for st in Stations for os in OilSpills}
        penalty = 0  # it is coming from model (so, pre-calculation may not make sense)

        #%% Calculate distance and response time

        coordinates_st, coordinates_st_dict = custom_func.extract_station_coordinate(station_data)
        coordinates_os, coordinates_spill_dict = custom_func.extract_spill_coordinate(spill_data)

        Distance = {(o, s): custom_func.compute_distance(coordinates_spill_dict[o], coordinates_st_dict[s])
                    for o in OilSpills for s in Stations }

        TimeR = Distance.copy()

        # %% Cost related
        Cf_data = input_parameters[['Station_cf', 'Cf_setup ($)']]
        Cf_s = {}
        for ii in Stations:
            Cf_s[ii] = Cf_data.loc[Cf_data['Station_cf'] == ii, 'Cf_setup ($)'].iloc[0]  # Cf_data[ii]

        # deploying mcr Distance*1 , deploying cdu Distance*2, deploying isb , Distance*3
        CostU = {}
        for key in Eff:
            CostU[key] = Eff[key] * 121

        return Stations, OilSpills, ResourcesD, Demand, Availability, Eff, Distance, TimeR, Cf_s, CostU

    def create_data_weight_pcp(self):
        spill_data = pd.read_excel('../data/processed/data_100_oil_spills.xlsx', sheet_name='spills', header=0).copy()
        potential_station_data = pd.read_excel('../data/raw/data_oil_spill_resource_allocation_Arctic_2023.xlsx',
                                               sheet_name='stations', header=0).copy()
        potential_station_data = potential_station_data.iloc[
            [0, 4, 7, 10, 11, 18]]  # if we want to keep current facility setup in the set
        station_data = pd.read_excel('../data/raw/data_oil_spill_resource_allocation_Arctic_2023.xlsx',
                                     sheet_name='stations', header=0).copy()
        current_station_data = pd.read_excel('../data/raw/data_oil_spill_resource_allocation_Arctic_2023.xlsx',
                                             sheet_name='current', header=0).copy()

        current_input_param = pd.read_excel('../data/raw/data_oil_spill_resource_allocation_Arctic_2023.xlsx',
                                            sheet_name='current input param', header=0).copy()
        input_parameters = pd.read_excel('../data/raw/data_oil_spill_resource_allocation_Arctic_2023.xlsx',
                                         sheet_name='Estimated parameters', header=0).copy()
        sensitivity_dataR = gpd.read_file('../data/gis map/ArcGIS_data/Sensitivity_data/Sensitivity_data5.shp').copy()

        coordinates_potential_st = custom_func.extract_station_coordinate(potential_station_data)
        potential_st_coord = pd.DataFrame(coordinates_potential_st[1]).T.reset_index().set_index('index')
        potential_st_coord.columns = ['St_Latitude', 'St_Longitude']
        coordinates_spill = custom_func.extract_spill_coordinate(spill_data)
        spill_coord = pd.DataFrame(coordinates_spill[1]).T.reset_index().set_index('index')
        spill_coord.columns = ['St_Latitude', 'St_Longitude']
        coordinates_st = custom_func.extract_station_coordinate(station_data)

        # Input param
        Stations, OilSpills, ResourcesD, Demand, Availability, Eff, Distance, TimeR, Cf_s, CostU \
            = data_preparation.generate_input_data(potential_station_data, spill_data, input_parameters)
        Distance_n = dict(zip(Distance.keys(),
                              custom_func.normalize(Distance.values(), min(Distance.values()), max(Distance.values()))))
        Effectiveness_n = dict(
            zip(Eff.keys(), custom_func.normalize(Eff.values(), min(Eff.values()), max(Eff.values()))))
        SizeSpill = SizeSpill_R = list(spill_data['Spill size']).copy()
        Sensitivity_R = custom_func.calculate_sensitivity(coordinates_spill[0], sensitivity_dataR)
        SizeSpill_n = dict(zip(OilSpills, custom_func.normalize(SizeSpill_R, min(SizeSpill_R), max(SizeSpill_R))))
        Sensitivity_n = dict(
            zip(OilSpills, custom_func.normalize(Sensitivity_R, min(Sensitivity_R), max(Sensitivity_R))))

        W = [100 * 0.25, 0.25, 100 * 0.25, 10 ** -4 * 0.25, 10 ** -2 * 0.25, 10 * 0.25]
        # we should emphasize spill size as well when trying to min distance. otherwise, it will minimize distance, & just assign nothing
        QuantityMin = 2
        DistanceMaxL = [5, 7, 8, 10, 11, 12, 15, 20]
        NumberStMaxL = [5, 8, 10, 12]
        select_1sL = []
        coverage_percentageL = []
        MeanResponseTimeL = []
        max_distance_sensitivity_L = []
        pareto_front_max_obj = []
        pareto_front_min_obj = []

        NumberStMaxL = [15, 12, 8, 5, 4, 2]
        number_of_spillsL = [200, 200, 200, 200, 200, 200]
        select_1sL = []
        coverage_percentageL = []
        MeanResponseTimeL = []
        DistanceMax = 10
        NumberStMax = 5
        BigM = 10 ** 20
        current_vs_proposed = 'proposed'  # current
        MaxFO = 1
        Budget = 10 ** 12
        W = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
        QuantityMin = 2

        # W = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
        # w1 = [.25, .5, .75]; w2 = [.25, .5, .75]; w3 = [.25, .5, .75];
        w1 = [.25, 10, 100];
        w2 = [10 ** -2, .25, 10, 100, 10 ** 4];
        w3 = [10 ** -2, .25, 1, 10, 100]
        w4 = [.25, .75];
        w5 = .25;
        w6 = .25;
        w7 = .25;
        w8 = .25

        weight_sensi_L = []
        MeanResponseTime_m4bL = []
        coverage_percentage_m4bL = []
        objValues_m4bL = []

        # Model
        for i in range(len(DistanceMaxL)):
            for j in range(len(NumberStMaxL)):
                model_04b, select_m4b, deploy_m4b, mvars_m4b, names_m4b, values_m4b, objValues_m4b, \
                    spill_df_m4b, station_df_m4b, cover_1s_m4b, select_1s_m4b, deploy_1s_m4b, MeanResponseTime_m4b, coverage_percentage_m4b, assignment_m4b \
                    = ModelW.solve(Stations, OilSpills, ResourcesD, coordinates_st, coordinates_spill, SizeSpill,
                                   SizeSpill_n,
                                   Demand, Sensitivity_R, Sensitivity_n, Eff, Effectiveness_n, Availability,
                                   NumberStMaxL[j], Distance, Distance_n,
                                   W, QuantityMin, DistanceMaxL[i], Cf_s, CostU, Budget,
                                   BigM, MaxFO)
                select_1sL.append(select_1s_m4b.index)
                coverage_percentageL.append(coverage_percentage_m4b)
                MeanResponseTimeL.append(MeanResponseTime_m4b)
                max_distance_sensitivity_L.append([DistanceMaxL[i], NumberStMaxL[j], MeanResponseTime_m4b])
                """
                assignment_line_m4b = DrawNetworkDiagram.draw_network_diagram(DistanceMaxL[i], NumberStMaxL[j], Sensitivity_R, spill_df_m4b, station_df_m4b, MeanResponseTime_m4b,      coverage_percentage_m4b, assignment_m4b, deploy_1s_m4b, select_1s_m4b, current_vs_proposed)
                """

        # Model
        for i in range(len(w1)):
            for j in range(len(w2)):
                for k in range(len(w3)):
                    for n in range(len(w4)):
                        W = [w1[i], w2[j], w3[k], w4[n], w5, w6, w7, w8]
                        model_04b, select_m4b, deploy_m4b, mvars_m4b, names_m4b, values_m4b, objValues_m4b, \
                            spill_df_m4b, station_df_m4b, cover_1s_m4b, select_1s_m4b, deploy_1s_m4b, MeanResponseTime_m4b, coverage_percentage_m4b, assignment_m4b \
                            = Model.solve(Stations, OilSpills, ResourcesD, coordinates_potential_st, coordinates_spill,
                                          SizeSpill, SizeSpill_n,
                                          Demand, Sensitivity_R, Sensitivity_n, Eff, Effectiveness_n, Availability,
                                          NumberStMax, Distance, Distance_n,
                                          W, QuantityMin, DistanceMax, Cf_s, CostU, Budget,
                                          BigM, MaxFO)

                        weight_sensi_L.append([w1[i], w2[j], w3[k], w4[n], w5, w6, w7, w8,
                                               coverage_percentage_m4b, MeanResponseTime_m4b,
                                               objValues_m4b[1::2][1]])  # coverage of sensitive area ++
        data_weight_pcp = pd.DataFrame(weight_sensi_L)
        data_weight_pcp.columns = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'coverage_percentage',
                                   'MeanResponseTime', 'objValues2']
        data_weight_pcp


    def create_boxplot_data(self):
        # read csv file for newly created 10 sets
        spill_data = pd.read_csv('../data/processed/10 Sets of spills/setA_data_100_oil_spills.csv')
        potential_station_data = pd.read_excel(
            '../data/processed/10 Sets of spills/data_oil_spill_resource_allocation_Arctic_2023_setA.xlsx',
            sheet_name='stations', header=0).copy()
        input_parameters = pd.read_excel(
            '../data/processed/10 Sets of spills/data_oil_spill_resource_allocation_Arctic_2023_setA.xlsx',
            sheet_name='Estimated parameters', header=0).copy()
        # %%
        # Data Extraction
        coordinates_potential_st = custom_func.extract_station_coordinate(potential_station_data)
        potential_st_coord = pd.DataFrame(coordinates_potential_st[1]).T.reset_index().set_index('index')
        potential_st_coord.columns = ['St_Latitude', 'St_Longitude']

        coordinates_spill = custom_func.extract_spill_coordinate(spill_data)
        spill_coord = pd.DataFrame(coordinates_spill[1]).T.reset_index().set_index('index')
        spill_coord.columns = ['St_Latitude', 'St_Longitude']

        # Input param
        Stations, OilSpills, ResourcesD, Demand, Availability, Eff, Distance, TimeR, Cf_s, CostU \
            = data_preparation.generate_input_data(potential_station_data, spill_data, input_parameters)

        SizeSpill = SizeSpill_R = list(spill_data['Spill size']).copy()
        Sensitivity_R = custom_func.calculate_sensitivity(coordinates_spill[0], sensitivity_dataR)
        # normalize
        SizeSpill_n = dict(zip(OilSpills, custom_func.normalize(SizeSpill_R, min(SizeSpill_R), max(SizeSpill_R))))
        Sensitivity_n = dict(
            zip(OilSpills, custom_func.normalize(Sensitivity_R, min(Sensitivity_R), max(Sensitivity_R))))

        Distance_n = dict(zip(Distance.keys(),
                              custom_func.normalize(Distance.values(), min(Distance.values()), max(Distance.values()))))
        # (x_0-x_min)/(x_max-x_min)
        Effectiveness_n = dict(
            zip(Eff.keys(), custom_func.normalize(Eff.values(), min(Eff.values()), max(Eff.values()))))
        # %% md

        # %%
        NumberStMax_m4b = 5
        DistanceMax = 10
        current_vs_proposed = 'proposed'
        # %%
        # Model
        model_04b, select_m4b, deploy_m4b, mvars_m4b, names_m4b, values_m4b, objValues_m4b, \
            spill_df_m4b, station_df_m4b, cover_1s_m4b, select_1s_m4b, deploy_1s_m4b, MeanResponseTime_m4b, coverage_percentage_m4b, assignment_m4b \
            = Model.solve(Stations, OilSpills, ResourcesD, coordinates_potential_st, coordinates_spill, SizeSpill,
                          SizeSpill_n,
                          Demand, Sensitivity_R, Sensitivity_n, Eff, Effectiveness_n, Availability, NumberStMax_m4b,
                          Distance, Distance_n,
                          W, QuantityMin, DistanceMax, Cf_s, CostU, Budget,
                          BigM, MaxFO)
        # %%
        # Draw the network
        assignment_line_m4b = DrawNetworkDiagram.draw_network_diagram(DistanceMax, NumberStMax_m4b, Sensitivity_R,
                                                                      spill_df_m4b, station_df_m4b,
                                                                      MeanResponseTime_m4b, coverage_percentage_m4b,
                                                                      assignment_m4b, deploy_1s_m4b, select_1s_m4b,
                                                                      current_vs_proposed)
        # %%

        # %%
        boxplot_df = pd.DataFrame(index=range(10),
                                  columns=['Set of 100 spills', 'Coverage (%)', 'Mean Respone Time',
                                           'Cost Objective value'])
        boxplot_df
        # %% md
        ## Run without for loop
        # %%
        # Load data
        # read csv file for newly created 10 sets
        spill_data = pd.read_csv('../data/processed/10 Sets of spills/setJ_data_100_oil_spills.csv')
        potential_station_data = pd.read_excel(
            '../data/processed/10 Sets of spills/data_oil_spill_resource_allocation_Arctic_2023_setJ.xlsx',
            sheet_name='stations', header=0).copy()
        input_parameters = pd.read_excel(
            '../data/processed/10 Sets of spills/data_oil_spill_resource_allocation_Arctic_2023_setJ.xlsx',
            sheet_name='Estimated parameters', header=0).copy()

        # Data Extraction
        coordinates_potential_st = custom_func.extract_station_coordinate(potential_station_data)
        potential_st_coord = pd.DataFrame(coordinates_potential_st[1]).T.reset_index().set_index('index')
        potential_st_coord.columns = ['St_Latitude', 'St_Longitude']
        coordinates_spill = custom_func.extract_spill_coordinate(spill_data)
        spill_coord = pd.DataFrame(coordinates_spill[1]).T.reset_index().set_index('index')
        spill_coord.columns = ['St_Latitude', 'St_Longitude']

        # Input param
        Stations, OilSpills, ResourcesD, Demand, Availability, Eff, Distance, TimeR, Cf_s, CostU \
            = data_preparation.generate_input_data(potential_station_data, spill_data, input_parameters)

        SizeSpill = SizeSpill_R = list(spill_data['Spill size']).copy()
        Sensitivity_R = custom_func.calculate_sensitivity(coordinates_spill[0], sensitivity_dataR)
        # normalize
        SizeSpill_n = dict(zip(OilSpills, custom_func.normalize(SizeSpill_R, min(SizeSpill_R), max(SizeSpill_R))))
        Sensitivity_n = dict(
            zip(OilSpills, custom_func.normalize(Sensitivity_R, min(Sensitivity_R), max(Sensitivity_R))))

        Distance_n = dict(zip(Distance.keys(),
                              custom_func.normalize(Distance.values(), min(Distance.values()), max(Distance.values()))))
        # (x_0-x_min)/(x_max-x_min)
        Effectiveness_n = dict(
            zip(Eff.keys(), custom_func.normalize(Eff.values(), min(Eff.values()), max(Eff.values()))))

        NumberStMax_m4b = 5
        DistanceMax = 10
        current_vs_proposed = 'proposed'

        # Model
        model_04b, select_m4b, deploy_m4b, mvars_m4b, names_m4b, values_m4b, objValues_m4b, \
            spill_df_m4b, station_df_m4b, cover_1s_m4b, select_1s_m4b, deploy_1s_m4b, MeanResponseTime_m4b, coverage_percentage_m4b, assignment_m4b \
            = Model.solve(Stations, OilSpills, ResourcesD, coordinates_potential_st, coordinates_spill, SizeSpill,
                          SizeSpill_n,
                          Demand, Sensitivity_R, Sensitivity_n, Eff, Effectiveness_n, Availability, NumberStMax_m4b,
                          Distance, Distance_n,
                          W, QuantityMin, DistanceMax, Cf_s, CostU, Budget,
                          BigM, MaxFO)

        SetName = 'Set J'
        if SetName == 'Set J':
            i = 9
            boxplot_df.iloc[i, 0] = SetName
            boxplot_df.iloc[i, 1] = coverage_percentage_m4b
            boxplot_df.iloc[i, 2] = MeanResponseTime_m4b
            boxplot_df.iloc[i, 3] = objValues_m4b[1::2][0]
        boxplot_df
        # %%
        boxplot_df.to_csv('../data/processed/boxplot_data_proposed.csv')
        # %% md
        ## Current facility location - model 3 (for 10 dataset) using for loop
        # %%
        boxplot_df_model3 = pd.DataFrame(index=range(10),
                                         columns=['Set of 100 spills', 'Coverage (%)', 'Mean Respone Time',
                                                  'Cost Objective value'])
        # %%
        SetNameL = ['setA', 'setB', 'setC', 'setD', 'setE', 'setF', 'setG', 'setH', 'setI', 'setJ']

        NumberStMax_m3 = 6
        DistanceMax = 10
        current_vs_proposed = 'current'

        for SetName in SetNameL:
            # Load data
            # read csv file for newly created 10 sets
            spill_data = pd.read_csv(f'../data/processed/10 Sets of spills/{SetName}_data_100_oil_spills.csv')

            coordinates_spill = custom_func.extract_spill_coordinate(spill_data)
            spill_coord = pd.DataFrame(coordinates_spill[1]).T.reset_index().set_index('index')
            spill_coord.columns = ['St_Latitude', 'St_Longitude']

            # Input param
            Stations, OilSpills, ResourcesD, Demand, Availability, Eff, Distance, TimeR, Cf_s, CostU \
                = data_preparation.generate_input_data(current_station_data, spill_data, current_input_param)

            SizeSpill = SizeSpill_R = list(spill_data['Spill size']).copy()
            Sensitivity_R = custom_func.calculate_sensitivity(coordinates_spill[0], sensitivity_dataR)
            # normalize
            SizeSpill_n = dict(zip(OilSpills, custom_func.normalize(SizeSpill_R, min(SizeSpill_R), max(SizeSpill_R))))
            Sensitivity_n = dict(
                zip(OilSpills, custom_func.normalize(Sensitivity_R, min(Sensitivity_R), max(Sensitivity_R))))

            Distance_n = dict(zip(Distance.keys(), custom_func.normalize(Distance.values(), min(Distance.values()),
                                                                         max(Distance.values()))))
            # (x_0-x_min)/(x_max-x_min)
            Effectiveness_n = dict(
                zip(Eff.keys(), custom_func.normalize(Eff.values(), min(Eff.values()), max(Eff.values()))))

            # Modeling based on current setup
            model_03, select_m3, deploy_m3, mvars_m3, names_m3, values_m3, objValues_m3, \
                spill_df_m3, station_df_m3, cover_1s_m3, select_1s_m3, deploy_1s_m3, MeanResponseTime_m3, coverage_percentage_m3, assignment_m3 \
                = Model.solve(Stations, OilSpills, ResourcesD, coordinates_st, coordinates_spill, SizeSpill,
                              SizeSpill_n,
                              Demand, Sensitivity_R, Sensitivity_n, Eff, Effectiveness_n, Availability, NumberStMax_m3,
                              Distance, Distance_n,
                              W, QuantityMin, DistanceMax, Cf_s, CostU, Budget,
                              BigM, MaxFO)

            if SetName == 'setA':
                i = 0
            elif SetName == 'setB':
                i = 1
            elif SetName == 'setC':
                i = 2
            elif SetName == 'setD':
                i = 3
            elif SetName == 'setE':
                i = 4
            elif SetName == 'setF':
                i = 5
            elif SetName == 'setG':
                i = 6
            elif SetName == 'setH':
                i = 7
            elif SetName == 'setI':
                i = 8
            elif SetName == 'setJ':
                i = 9
            boxplot_df_model3.iloc[i, 0] = SetName
            boxplot_df_model3.iloc[i, 1] = coverage_percentage_m3
            boxplot_df_model3.iloc[i, 2] = MeanResponseTime_m3
            boxplot_df_model3.iloc[i, 3] = objValues_m3[1::2][0]

        boxplot_df_model3
        # %%
        boxplot_df_model3.to_csv('../data/processed/boxplot_df_model3.csv')