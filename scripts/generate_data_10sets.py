#
import pandas as pd
import geopandas as gpd
import pickle
from src.preprocessing import preprocess_utils, data_loader
from src.config import config_loader

# Load config
config = config_loader.load_config("../src/config/model_config.yaml")

# Load static sensitivity data (same for all models)
sensitivity_data = gpd.read_file("../data/gis map/ArcGIS_data/Sensitivity_data/Sensitivity_data5.shp")

station_type = [{"name": "proposed",
                 "spill_data_path": "../data/spill data/oil_spills_100_data.xlsx",
                 "station_sheet": "stations",
                 "param_sheet": "Estimated parameters",
                 "station_rows": None},
                {"name": "current",
                 "spill_data_path": "../data/spill data/oil_spills_100_data.xlsx",  # Same spill data (update if needed)
                 "station_sheet": "current",
                 "param_sheet": "current input param",
                 "station_rows": None}]

for SetName in ['setA', 'setB', 'setC', 'setD', 'setE', 'setF', 'setG', 'setH', 'setI', 'setJ']:
    for stationType in station_type:
        # Load data
        spill_data = pd.read_csv(f'../data/10 Sets of spills/{SetName}_data_100_oil_spills.csv')
        print('stationType["station_sheet"]: ', stationType["station_sheet"], stationType["param_sheet"])
        station_data = pd.read_excel("../data/spill data/data_oil_spill_resource_allocation_Arctic_2023.xlsx",
                                     sheet_name=stationType["station_sheet"])
        parameters = pd.read_excel("../data/spill data/data_oil_spill_resource_allocation_Arctic_2023.xlsx",
                                   sheet_name=stationType["param_sheet"])

        # Preprocess
        Stations, OilSpills, Resources, demand_or, demand_ov, A_sr, Eff_sor, Distance, t_os, F_s = (
            preprocess_utils.generate_input_data(station_data, spill_data, parameters))

        coordinates_spill = preprocess_utils.extract_spill_coordinate(spill_data)
        coordinates_st = preprocess_utils.extract_station_coordinate(station_data)
        SizeSpill = list(spill_data['Spill size'])
        Sensitivity_R = preprocess_utils.calculate_sensitivity(list(coordinates_spill.values()), sensitivity_data)

        v_o_n = dict(zip(OilSpills, preprocess_utils.normalize(SizeSpill, min(SizeSpill), max(SizeSpill))))
        eta_o = dict(zip(OilSpills, preprocess_utils.normalize(Sensitivity_R, min(Sensitivity_R), max(Sensitivity_R))))
        Distance_n = dict(zip(Distance.keys(), preprocess_utils.normalize(Distance.values(), min(Distance.values()),
                                                                          max(Distance.values()))))
        t_os_n = dict(zip(t_os.keys(), preprocess_utils.normalize(t_os.values(), min(t_os.values()), max(t_os.values()))))
        pn_sor = preprocess_utils.compute_penalty(Stations, OilSpills, Resources)

        # Save .pkl
        preprocessed_data = {
            "Stations": Stations, "OilSpills": OilSpills, "Resources": Resources,
            "demand_or": demand_or, "demand_ov": demand_ov, "A_sr": A_sr, "Eff_sor": Eff_sor,
            "Distance": Distance, "t_os": t_os, "F_s": F_s,
            "coordinates_spill": coordinates_spill, "coordinates_st": coordinates_st,
            "Sensitivity_R": Sensitivity_R,
            "v_o": SizeSpill,
            "v_o_n": v_o_n, "eta_o": eta_o, "Distance_n": Distance_n,
            "t_os_n": t_os_n, "pn_sor": pn_sor
        }
        with open(f"../data/10 Sets of spills/preprocessed_data_{SetName}_{stationType['name']}.pkl", "wb") as f:
            pickle.dump(preprocessed_data, f)

print('Data generation completed!')
