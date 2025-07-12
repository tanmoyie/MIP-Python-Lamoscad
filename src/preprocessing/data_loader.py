
"""
data_loader.py
Purpose: Load raw input data from files (Excel, shapefile, etc.)
"""

import pandas as pd
import geopandas as gpd

def load_spill_data(filepath):
    """Load oil spill data from Excel."""
    return pd.read_excel(filepath)

def load_station_data(filepath):
    """Load station data from Excel."""
    return pd.read_excel(filepath)

def load_parameters(filepath):
    """Load model parameters from Excel."""
    return pd.read_excel(filepath)

def load_sensitivity_data(filepath):
    """Load sensitivity data from shapefile."""
    return gpd.read_file(filepath)
