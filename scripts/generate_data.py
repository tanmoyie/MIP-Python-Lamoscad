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
models = [
    {
        "name": "model_p256",
        "spill_data_path": "../data/spill data/oil_spills_100_data.xlsx",
        "station_sheet": "stations",
        "param_sheet": "Estimated parameters",
        "station_rows": None  # Use all rows
    },
    {
        "name": "model_c",
        "spill_data_path": "../data/spill data/oil_spills_100_data.xlsx",  # Same spill data (update if needed)
        "station_sheet": "current",
        "param_sheet": "current input param",
        "station_rows": None
    },
    {
        "name": "model_3",
        "spill_data_path": "../data/spill data/oil_spills_100_data.xlsx",
        "station_sheet": "stations",
        "param_sheet": "Estimated parameters",
        "station_rows": [0, 4, 7, 10, 11, 18]
    }
]

# === Loop over models ===
for model in models:
    print(f"\nProcessing {model['name']}...")

    # Load data
    spill_data = pd.read_excel(model['spill_data_path'])
    station_data = pd.read_excel("../data/spill data/data_oil_spill_resource_allocation_Arctic_2023.xlsx",
                                 sheet_name=model["station_sheet"])
    parameters = pd.read_excel("../data/spill data/data_oil_spill_resource_allocation_Arctic_2023.xlsx",
                               sheet_name=model["param_sheet"])

    if model["station_rows"]:
        station_data = station_data.iloc[model["station_rows"]]

    # Preprocess
    Stations, OilSpills, Resources, demand_or, demand_ov, A_sr, Eff_sor, Distance, t_os, F_s, C_sr = (
        preprocess_utils.generate_input_data(station_data, spill_data, parameters))

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
        "Distance": Distance, "t_os": t_os, "F_s": F_s, "C_sr": C_sr,
        "coordinates_spill": coordinates_spill, "coordinates_st": coordinates_st,
        "SizeSpill": SizeSpill, "Sensitivity_R": Sensitivity_R,
        "v_o_n": v_o_n, "eta_o": eta_o, "Distance_n": Distance_n,
        "t_os_n": t_os_n, "pn_sor": pn_sor
    }
    with open(f"../data/preprocessed_data_{model['name']}.pkl", "wb") as f:
        pickle.dump(preprocessed_data, f)

    # Save .xlsx
    if model['name'] == 'model_p256':
        with pd.ExcelWriter(f"../data/preprocessed_data_{model['name']}.xlsx") as writer:
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
            pd.DataFrame.from_dict(C_sr, orient="index").to_excel(writer, sheet_name="C_sr")
            pd.DataFrame.from_dict(Eff_sor, orient="index").to_excel(writer, sheet_name="Eff_sor")
            pd.DataFrame.from_dict(pn_sor, orient="index").to_excel(writer, sheet_name="pn_sor")

    print(f"✅ Saved preprocessed data for {model['name']}.")
