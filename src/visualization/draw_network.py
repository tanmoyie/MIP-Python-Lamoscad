""" Draw Network diagram

Developer: Tanmoy Das
"""

# import libraries
from src.preprocessing.preprocess_utils import compute_distance
import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
""" Plot Network Diagram """


def draw_network_diagram(y_os1, spill_df, station_df,
                         name, proposed=True):

    if proposed:
        station_color = 'yellow'
        shape = 's'
        alpha_val = 1
        facility_size = 400
        legend_y_position = 0.25
        facility_text_color = 'red'
        speed = 30
    else:
        station_color = 'green'
        shape = '^'
        alpha_val = 1
        facility_size = 700
        legend_y_position = 0.50
        facility_text_color = 'green'
        st_name = ['$s_{tu}$', '$s_{ha}$', '$s_{ch}$', '$s_{iq}$']

    # ............................. Extract related data
    selected_supply_stations = list(y_os1.reset_index().level_1.unique())
    spill_df_covered = spill_df[spill_df['Spill #'].isin([item[0] for item in y_os1.index])]
    spill_df_not_covered = spill_df[~spill_df['Spill #'].isin([item[0] for item in y_os1.index])]
    station_df_selected = station_df[station_df['Station no.'].isin(selected_supply_stations)]

    # ............................. Draw the GIS map
    fig, ax = plt.subplots(figsize=(6, 6))  # ++
    # Load geometric file for map
    ArcticMap = gpd.read_file("../data/gis map/ArcGIS_data/ArcticShapefile2/ArcticShapefile2.shp")
    ArcticMap = ArcticMap.to_crs(epsg=4326)  # 3857
    ArcticLand = ArcticMap[ArcticMap['Region_EN'].isin(['Arctic'])]
    ArcticWater = ArcticMap[ArcticMap['Region_EN'].isin(['Arctic-Water'])]
    ArcticLand.plot(ax=ax, color="seashell")
    ArcticWater.plot(ax=ax, color="lightskyblue")

    # ............................. Plotting oil spills, response stations
    # Normalizing the size of circle (oil spill) in the graph so that even if spill size is 80000,
    x_max = max(spill_df[spill_df['Spill #'].isin([item[0] for item in y_os1.index])]['Spill Size'])
    x_min = min(spill_df[spill_df['Spill #'].isin([item[0] for item in y_os1.index])]['Spill Size'])

    # Facecolor of ALL spills
    spillAll = plt.scatter(data=spill_df, x='Spill_Longitude', y='Spill_Latitude',
                           s=((spill_df['Spill Size'])
                              - x_min) * 500 / (x_max - x_min),
                           c=(spill_df['Sensitivity']),
                           cmap='binary')

    # Points of covered spills
    spillC = plt.scatter(data=spill_df_covered,
                         x='Spill_Longitude', y='Spill_Latitude',
                         s=((spill_df_covered[
                             'Spill Size'])
                            - x_min) * 500 / (x_max - x_min),
                         facecolors='none', edgecolors='green', alpha=0.4)

    # if there is any uncover spill, run the below conditional code block
    if len(spill_df[~spill_df['Spill #'].isin([item[0] for item in y_os1.index])]['Spill Size']) > 0:

        # Edge of un-covered spills
        x1_max = max(spill_df[~spill_df['Spill #'].isin([item[0] for item in y_os1.index])]['Spill Size'])
        x1_min = min(spill_df[~spill_df['Spill #'].isin([item[0] for item in y_os1.index])]['Spill Size'])
        spillUnC = plt.scatter(data=spill_df_not_covered,
                               x='Spill_Longitude', y='Spill_Latitude',
                               s=((spill_df_not_covered[
                                   'Spill Size'])
                                  - x1_min) * 500 / (x1_max - x1_min),
                               facecolors='none', edgecolors='red', alpha=0.5)
    else:
        spillUnC = 0

    # Plot stations (squares)
    st = plt.scatter(data=station_df_selected,
                     x='St_Longitude', y='St_Latitude',
                     marker=shape, alpha=alpha_val, s=facility_size,
                     edgecolors='black',
                     zorder=5,
                     c=station_color)
    # Annotate station numbers
    for ii, row in station_df_selected.iterrows():
        ax.text(row['St_Longitude']+1.1, row['St_Latitude']-0.3, f"$s_{{{ii + 1}}}$",
                zorder=6, fontsize=10, ha='right', va='bottom', color=facility_text_color, fontweight='bold')

    # Showing station number as text for current stations
    if proposed is False:
        for i in range(len(selected_supply_stations)):
            plt.text(station_df_selected.St_Longitude[i] - 1.5, y=station_df_selected.St_Latitude[i] - legend_y_position,
                     zorder=6, s=st_name[i], fontdict=dict(color='white', size=10))

    # Small purple squares to show non-selected stations
    stUns = plt.scatter(data=station_df[~station_df['Station no.'].isin(selected_supply_stations)],
                        x='St_Longitude', y='St_Latitude', marker='s', alpha=.25, c='blue')

    # Get unique stations and assign colors
    unique_stations = list(station_df_selected['Station no.'].unique())
    cmap = cm.get_cmap("tab10", len(unique_stations))  # Using a colormap with distinct colors
    station_colors = {station: mcolors.to_hex(cmap(i)) for i, station in enumerate(unique_stations)}

    # Draw lines between eligible pairs with different colors for different stations
    total_response_time = 0
    for spill, station in list(y_os1.index):
        spill_loc = spill_df[spill_df['Spill #'] == spill]
        station_loc = station_df[station_df['Station no.'] == station]

        if not spill_loc.empty and not station_loc.empty:
            x_values = [spill_loc.iloc[0]['Spill_Longitude'], station_loc.iloc[0]['St_Longitude']]
            y_values = [spill_loc.iloc[0]['Spill_Latitude'], station_loc.iloc[0]['St_Latitude']]
            total_response_time += compute_distance(x_values, y_values)

            # Use assigned color for each station
            line_color = station_colors.get(station, 'k')  # Default to black if station not found
            ax.plot(x_values, y_values, '-', alpha=0.8, color=line_color)

    plt.legend((spillC, spillUnC, st, stUns),
                               ('Oil Spill covered', 'Oil Spill not covered', 'Station selected',
                                'Station not selected'),
                               loc='lower left', ncol=1, handlelength=5, borderpad=.1, markerscale=.4,
                               fontsize=10)

    ax.set_xlim([-141, -60])
    ax.set_ylim([51, 84])
    plt.axis('off')
    fig.tight_layout()
    plt.savefig(f'../results/plots/network_diagram_{name}.png', dpi=500)
    # plt.show()
    plt.close()
