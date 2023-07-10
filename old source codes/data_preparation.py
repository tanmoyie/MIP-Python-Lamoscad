""" File name: data_generation.py

Outline:
Calculating Demand, effectiveness, availability, distance, fixed and variable costs

Developer: Tanmoy Das
Date: March 2023

"""

import custom_func


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
    # ++ create data on Excel file (for now, directly to eff)
    CostU = {}
    for key in Eff:
        CostU[key] = Eff[key] * 121
    return Stations, OilSpills, ResourcesD, Demand, Availability, Eff, Distance, TimeR, Cf_s, CostU
