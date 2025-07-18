"""
File Name: custom_func.py


All the custom functions related to data processing and optimization modeling are written here
& called as module in main.ipynb or model.py script
Function list
1. Compute Distance
2. Compute Pairings
3. Compute Time to Respond
4. Extract Coordinate
5. Calculate Sensitivity

"""
# Compute distance
import math
import pandas as pd
from shapely.geometry import Point
import shapely
import geopandas as gpd
import random

# 1
def compute_distance(loc1, loc2):
    """
    Compute distance between two geolocations given their coordinates
    :param loc1: latitude and longitude of point 1
    :param loc2: latitude and longitude of point 2
    :return: Distance (possibly in kilometer) between two points  +++ great circle distance
    """
    dx = loc1[0] - loc2[0]
    dy = loc1[1] - loc2[1]
    return round(math.sqrt(dx * dx + dy * dy), 2)


def compute_pairing(coordinates_spill, coordinates_st, DistanceMax):
    """
    Compute the possible pairing of (o,s)
    :param coordinates_spill:
    :param coordinates_st:
    :param DistanceMax:
    :return:
    """
    pairings = {(c, f): compute_distance(coordinates_spill[c], coordinates_st[f])
                for c in range(len(coordinates_spill))
                for f in range(len(coordinates_st))
                # need to delete the below line later ++
                if compute_distance(tuple(coordinates_spill[c]), tuple(coordinates_st[f])) < DistanceMax}
    return pairings


def normalize(x_r, x_min, x_max):
    return [round((x_0-x_min)/(x_max-x_min), 2) for x_0 in x_r]


def compute_TimeR(pairings, spill_data):
    """

    :param pairings:
    :param spill_data:
    :return:
    """
    TimeR = pairings
    rank1 = spill_data[['1st Ranking']]
    for i in range(len(rank1)):
        if rank1[i] == "MCR":
            TimeR[i, :] = pairings[i, :]  # pairings values = distance
        elif rank1[i] == "CDU" or "ISB":
            TimeR[i, :] = pairings[i, :] / 10
    return TimeR


# Extract coordinates in right format
def extract_coordinate(data):
    """

    :param data:
    :return: two list inside a dictionary: List 1 just the coordinates; List 2 coordinates with Spill no. as index
    """
    # location of demands
    coordinates_in = data[['Coordinates']]  # .values.tolist()
    # preprocessing (what exactly?)
    temp_df2 = coordinates_in.Coordinates.str.split(",", expand=True, )
    temp_df2['Extracted_1'] = temp_df2[0].str.extract('([-+]?\d*\.?\d+)')
    temp_df2['Extracted_2'] = temp_df2[1].str.extract('([-+]?\d*\.?\d+)')
    temp_df2["Extracted_1"] = pd.to_numeric(temp_df2["Extracted_1"], downcast="float")
    temp_df2["Extracted_2"] = pd.to_numeric(temp_df2["Extracted_2"], downcast="float")
    # Getting coordinates of stations in a format needed for Folium MAP
    coordinates = temp_df2[['Extracted_1', 'Extracted_2']].values.tolist()
    coordinates_dict = {}
    for i in range(len(coordinates)):
        coordinates_dict[data.reset_index().at[i, 'Spill #']] = coordinates[i]
    return coordinates, coordinates_dict


# Extract coordinates in right format
def extract_spill_coordinate(data):
    # location of demands
    coordinates_in = data[['Coordinates']]  # .values.tolist()  Coordinates
    # preprocessing (what exactly?)
    temp_df2 = coordinates_in.Coordinates.str.split(",", expand=True, )
    temp_df2['Extracted_1'] = temp_df2[0].str.extract('([-+]?\d*\.?\d+)')
    temp_df2['Extracted_2'] = temp_df2[1].str.extract('([-+]?\d*\.?\d+)')
    temp_df2["Extracted_1"] = pd.to_numeric(temp_df2["Extracted_1"], downcast="float")
    temp_df2["Extracted_2"] = pd.to_numeric(temp_df2["Extracted_2"], downcast="float")
    # Getting coordinates of stations in a format needed for Folium MAP
    coordinates = temp_df2[['Extracted_1', 'Extracted_2']].values.tolist()
    coordinates_dict = {}
    for i in range(len(coordinates)):
        coordinates_dict[data.reset_index().at[i, 'Spill #']] = coordinates[i]
    return coordinates_dict


# Extract coordinates in right format
def extract_station_coordinate(data):
    # location of demands
    coordinates_in = data[['Coordinates']]  # .values.tolist()
    # preprocessing (what exactly?)
    temp_df2 = coordinates_in.Coordinates.str.split(",", expand=True, )
    temp_df2['Extracted_1'] = temp_df2[0].str.extract('([-+]?\d*\.?\d+)')
    temp_df2['Extracted_2'] = temp_df2[1].str.extract('([-+]?\d*\.?\d+)')
    temp_df2["Extracted_1"] = pd.to_numeric(temp_df2["Extracted_1"], downcast="float")
    temp_df2["Extracted_2"] = pd.to_numeric(temp_df2["Extracted_2"], downcast="float")
    # Getting coordinates of stations in a format needed for Folium MAP
    coordinates = temp_df2[['Extracted_1', 'Extracted_2']].values.tolist()
    coordinates_dict = {}
    for i in range(len(coordinates)):
        coordinates_dict[data.reset_index().at[i, 'Station #']] = coordinates[i]
    return coordinates_dict


def calculate_sensitivity(coordinates_spill, sensitivity_dataR):
    """
    :param coordinates_spill:
    :param sensitivity_dataR:
    :return:
    """
    G_series = sensitivity_dataR.geometry.map(lambda polygon: shapely.ops.transform(lambda x, y: (y, x), polygon))
    sensitivity_data = gpd.GeoDataFrame(geometry=gpd.GeoSeries(G_series))
    sensitivity_data['Sensitivity'] = sensitivity_dataR[['Sensitivit']]

    Sensitivity = []
    for i in range(len(coordinates_spill)):
        # Coordinate of spill zone i
        spill_zone_i = Point(
            coordinates_spill[i])  # demand_i_coord  coordinates_spill[i] # need to work on NAN in dataset
        # list comprehension to determine which sensitive area this spill belongs
        spill_zone_contains = [sensitivity_data.loc[g, 'geometry'].contains(spill_zone_i) for g in
                               range(len(sensitivity_data))]
        # Calculate sensitivity value of spill zone i
        try:
            SN_within1 = sensitivity_data.loc[spill_zone_contains.index(True), 'Sensitivity']  # +++
        except:
            SN_within1 = 0
        # Create a circle around spill zone i
        spill_zone_larger = spill_zone_i.buffer(10)  # 10 is fine??
        # Find all intersecting neighborhood of sensitive areas of spill zone i
        spill_zone_within_neighbor = [spill_zone_larger.intersects(sensitivity_data.loc[j, 'geometry'])
                                      for j in range(len(sensitivity_data))]
        index_neighbor = [nei for nei in range(len(spill_zone_within_neighbor)) if
                          spill_zone_within_neighbor[nei] == True]
        # Calculate total sensitivity value of neighborhood
        SN_neighbor = sum(sensitivity_data.loc[index_neighbor, 'Sensitivity'])
        # Total sensitivity value of spill i
        sensitivity_i = 10 * SN_within1 + SN_neighbor
        Sensitivity.append(sensitivity_i)
    return Sensitivity


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
    # print(Stations)
    OilSpills = list(spill_data['Spill #'].unique())
    print('len(OilSpills): ', len(OilSpills))
    ResourcesD = ['m', 'c', 'i']

    #%% Demand[o,r]
    Demand_df = input_parameters[['Spill #', 'm', 'c', 'i']]
    D_stacked_df = Demand_df.set_index('Spill #').stack()

    o_r_comb = [(o, r) for o in OilSpills for r in ResourcesD]
    Demand = {}
    for ii in range(len(o_r_comb)):
        Demand[(o_r_comb[ii])] = D_stacked_df[ii]

    # print('\nDemand\n', pd.DataFrame(Demand.items(), columns=['(spill, resource)', 'Demand']).head(5))
    # print('Demand_or: ', Demand)
    response_vehicles = ["helicopter", "ship", "icebreaker"]
    demand_ov = {(spill, vehicle): 0 for spill in OilSpills for vehicle in response_vehicles}
    for (spill, category), value in Demand.items():
        # Helicopter conditions
        if category == 'c' and value > 750 or category == 'i' and value > 1000:
            demand_ov[(spill, 'helicopter')] += 4
        elif category == 'c' and value > 400 or category == 'i' and value > 0:  # 500 previous value for 0
            demand_ov[(spill, 'helicopter')] += 2

        # Ship and Ice Breaker conditions
        if category == 'm' and value > 50:
            demand_ov[(spill, 'ship')] += 2
            demand_ov[(spill, 'icebreaker')] += 1
        elif category == 'm' and value > 30:
            demand_ov[(spill, 'ship')] += 1
            demand_ov[(spill, 'icebreaker')] += 1
    # print('demand_ov: ', demand_ov)
    # %% Effectiveness[s,r] need to add o later ++
    Eff_sor = {}
    def generate_value(category):
        if category == 'm':
            return round(random.uniform(0.2, 0.3), 2)
        elif category == 'i':
            return round(random.uniform(0.8, 0.9), 2)
        elif category == 'c':
            return round(random.uniform(0.5, 0.7), 2)

    # Update the dictionary with new values for each combination of ('s', 'o', category)
    for s in Stations:  # Define the spill sites
        for o in OilSpills:  # Define the oil spill types
            for category in ['m', 'i', 'c']:  # Define the categories (m, i, c)
                Eff_sor[(s, o, category)] = generate_value(category)

    #%% Availability of resources in a station
    Ava_df = input_parameters[['Station.1', 'm.2', 'c.2', 'i.2']]
    Ava_df.columns = ['Station', 'm', 'c', 'i']
    Ava_stacked_df = Ava_df.set_index('Station').stack()
    s_r_comb = {(s, r): None for s in Stations for r in ResourcesD}
    Availability = {s_r_comb: Ava for ((s_r_comb), Ava) in zip(s_r_comb, Ava_stacked_df)}
    # print('\nAvailability\n', pd.DataFrame(Availability.items(), columns=['(station, resources)', 'Amount']).head(5))

    #%%
    # TimeWindowM = 6 * 60 if Tech_index == 'cdu' else (3 * 60 if Tech_index == 'cdu' else TimeRMax)
    # TimeR = {(st, os): custom_func.compute_distance(coordinates_st[st], coordinates_spill[os])
    #         for st in Stations for os in OilSpills}
    penalty = 0  # it is coming from model (so, pre-calculation may not make sense)

    #%% Calculate distance and response time

    coordinates_st = extract_station_coordinate(station_data)
    coordinates_os = extract_spill_coordinate(spill_data)

    Distance = {(o, s): compute_distance(coordinates_os[o], coordinates_st[s])
                for o in OilSpills for s in Stations}
    # print('\nDistance\n', pd.DataFrame(Distance.items(), columns=['(spill, station)', 'Distance']).head(5))
    TimeR = {(o, s): round(i/10, 2) for (o, s), i in Distance.items()}  # assuming vessel speed of 10 km/hr

    # %% Cost related
    Cf_data = input_parameters[['Station_cf', 'Cf_setup ($)']]
    Cf_s = {}
    for ii in Stations:
        Cf_s[ii] = Cf_data.loc[Cf_data['Station_cf'] == ii, 'Cf_setup ($)'].iloc[0]  # Cf_data[ii]

    return Stations, OilSpills, ResourcesD, Demand, demand_ov, Availability, Eff_sor, Distance, TimeR, Cf_s


def compute_penalty(Stations, OilSpills, ResourcesD):
    sor_comb = [(s, o, r) for s in Stations for o in OilSpills for r in ResourcesD]
    penalty_sor = {}
    for i in range(len(sor_comb)):
        penalty_sor[(sor_comb[i])] = 1
    return penalty_sor


import pandas as pd


def create_spill_dataframe(coordinates_spill, SizeSpill, Sensitivity_R):
    """Create a formatted DataFrame for spill information."""
    df = pd.DataFrame(coordinates_spill).T.reset_index()
    df.columns = ['Spill #', 'Spill_Latitude', 'Spill_Longitude']
    df['Spill Size'] = pd.DataFrame(SizeSpill)
    df['Sensitivity'] = Sensitivity_R
    return df


def create_station_dataframe(coordinates_st):
    """Create a formatted DataFrame for station information."""
    df = pd.DataFrame(coordinates_st).T.reset_index()
    df.columns = ['Station no.', 'St_Latitude', 'St_Longitude']
    return df


import pandas as pd


def convert_table5(coverage_percentage, mean_response_time, sensitive_areas_covered, runtime, num_facilities, name):
    """
    Create a results summary table for the LAMOSCAD model (Table 5).

    Parameters:
    - coverage_percentage: float
    - mean_response_time: float
    - sensitive_areas_covered: int
    - runtime: float (seconds)
    - num_facilities: int

    Returns:
    - pandas.DataFrame with labeled metrics
    """
    data = {
        "Metric": [
            "Coverage Percentage (%)",
            "Mean Response Time (in hours)",
            "Total number of oil spills in the sensitive areas covered",
            "Runtime of the model (in seconds)",
            "Number of facilities selected"
        ],
        name: [
            coverage_percentage,
            mean_response_time,
            sensitive_areas_covered,
            runtime,
            num_facilities
        ]
    }
    return pd.DataFrame(data)
