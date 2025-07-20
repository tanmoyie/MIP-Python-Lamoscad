""" Generate large-scale data for testing"""
import pandas as pd
import geopandas as gpd
import pickle
from src.preprocessing import preprocess_utils, data_loader
from src.config import config_loader

# Load config
config = config_loader.load_config("../src/config/model_config.yaml")

# Load static sensitivity data (same for all models)
sensitivity_data = gpd.read_file("../data/gis map/ArcGIS_data/Sensitivity_data/Sensitivity_data5.shp")

# === Define model scenarios ===

# === Loop over instances ===
instance_o = [200, 300, 400, 500, 750, 1000]
instance_st = [10, 20, 30, 50]

rows = []
for oil in instance_o:
    for sta in instance_st:
        instance = f'data_o{oil}_s{sta}'
        rows.append(f"{oil} × {sta}")
        print(f"\nProcessing {instance}...")

        # Load data
        spill_data = pd.read_excel(f'../data/spill data/oil_spills_{oil}_data.xlsx')   # oil_spills_200_data
        station_data = pd.read_excel("../data/spill data/data_oil_spill_resource_allocation_Arctic_2023_large_scale.xlsx",
                                     sheet_name='stations', nrows=sta if sta != 10 else 20)
        parameters = pd.read_excel("../data/spill data/data_oil_spill_resource_allocation_Arctic_2023_large_scale.xlsx",
                                   sheet_name='Estimated parameters')

        # Preprocess
        Stations, OilSpills, Resources, demand_or, demand_ov, A_sr, Eff_sor, Distance, t_os, F_s = (
            preprocess_utils.generate_input_data(station_data, spill_data, parameters))
        print('len st: ', len(Stations))
        coordinates_spill = preprocess_utils.extract_spill_coordinate(spill_data)
        coordinates_st = preprocess_utils.extract_station_coordinate(station_data)
        SizeSpill = list(spill_data['Spill size'])
        Sensitivity_R = preprocess_utils.calculate_sensitivity(list(coordinates_spill.values()), sensitivity_data)

        v_o_n = dict(zip(OilSpills, preprocess_utils.normalize(SizeSpill, min(SizeSpill), max(SizeSpill))))
        eta_o = dict(zip(OilSpills, preprocess_utils.normalize(Sensitivity_R, min(Sensitivity_R), max(Sensitivity_R))))
        Distance_n = dict(zip(Distance.keys(), preprocess_utils.normalize(Distance.values(), min(Distance.values()), max(Distance.values()))))
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
        with open(f"../data/large scale processed dataset/preprocessed_data_o{oil}_s{sta}.pkl", "wb") as f:
            pickle.dump(preprocessed_data, f)

        # Save .xlsx
        if instance == 'data_o200_s10':
            with pd.ExcelWriter(f"../data/large scale processed dataset/preprocessed_data_o{oil}_s{sta}.xlsx") as writer:
                pd.DataFrame(Stations).to_excel(writer, sheet_name="Stations")
                pd.DataFrame(OilSpills).to_excel(writer, sheet_name="OilSpills")
                pd.DataFrame(Resources).to_excel(writer, sheet_name="Resources")
                pd.DataFrame.from_dict(coordinates_spill, orient="index").to_excel(writer, sheet_name="Coordinates_Spill")
                pd.DataFrame.from_dict(coordinates_st, orient="index").to_excel(writer, sheet_name="Coordinates_Station")
                pd.DataFrame.from_dict(Distance, orient="index").to_excel(writer, sheet_name="Distance")
                pd.DataFrame.from_dict(t_os, orient="index").to_excel(writer, sheet_name="t_os")
                pd.DataFrame.from_dict(v_o_n, orient="index").to_excel(writer, sheet_name="v_o")
                pd.DataFrame.from_dict(eta_o, orient="index").to_excel(writer, sheet_name="eta_o")
                pd.DataFrame.from_dict(F_s, orient="index").to_excel(writer, sheet_name="F_s")
                pd.DataFrame.from_dict(Eff_sor, orient="index").to_excel(writer, sheet_name="Eff_sor")
                pd.DataFrame.from_dict(pn_sor, orient="index").to_excel(writer, sheet_name="pn_sor")

