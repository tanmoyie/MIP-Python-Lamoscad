"""
File Name: model_analysis.py

Outline: Data visualization
1. Draw Network diagram
2.

Developer: Tanmoy Das
Date: March 2023
"""

# import libraries
import pandas as pd
import custom_func
from matplotlib import collections as mc
import matplotlib.pyplot as plt
import folium
import geopandas as gpd
import numpy as np


class DrawNetworkDiagram:
    """ Plot Network Diagram """
    def draw_network_diagram(DistanceMax, NumberStMax, Sensitivity_R, spill_df, station_df, ResponseTimeT,
                             coverage_percentage,
                             assignment, deploy_1s, select_1s, current_vs_proposed):
        """ Plot the line segments, incident points, and base station points of the final network
        :param DistanceMax:
        :param NumberStMax:
        :param spill_df:
        :param station_df:
        :param ResponseTimeT:
        :param coverage_percentage:
        :param assignment:
        :param deploy_1s:
        :param select_1s:
        :return: Network diagram
        """
        from datetime import datetime, date

        if current_vs_proposed == 'current':
            station_color = 'green'
            shape = '^'
            alpha_val = 1
            facility_size = 700
            legend_y_position = 0.50
            facility_text_color = 'green'
            st_name = ['$s_{tu}$', '$s_{ha}$', '$s_{ch}$', '$s_{iq}$']
        else:
            station_color = 'yellow'
            shape = 's'
            alpha_val = 1
            facility_size = 400
            legend_y_position = 0.25
            facility_text_color = 'red'

        fig, ax = plt.subplots(figsize=(6, 6))  # ++
        # Load geometric file for map
        ArcticMap = gpd.read_file("../data/gis map/ArcGIS_data/ArcticShapefile2/ArcticShapefile2.shp")
        ArcticMap = ArcticMap.to_crs(epsg=4326)  # 3857
        ArcticLand = ArcticMap[ArcticMap['Region_EN'].isin(['Arctic'])]
        ArcticWater = ArcticMap[ArcticMap['Region_EN'].isin(['Arctic-Water'])]
        ArcticLand.plot(ax=ax, color="seashell")
        ArcticWater.plot(ax=ax, color="lightskyblue")

        # Drawing lines/ edges
        assignment_line = pd.DataFrame(columns=list(assignment.columns.values))
        for i, o in enumerate(assignment['Spill #'].unique()):
            minD = min(assignment.loc[assignment['Spill #'] == o]['Distance'])
            df = assignment.iloc[np.where(assignment['Distance'] == minD)[0]]
            assignment_line = pd.concat([assignment_line, df])

        unique_stations = assignment_line['Station no.'].unique()

        for ust in range(len(unique_stations)):
            d1 = assignment_line.loc[assignment_line['Station no.'] == unique_stations[ust]].reset_index()
            new_list = []
            for r in range(d1.shape[0]):
                new_list.append(
                    [(d1.Spill_Longitude[r], d1.Spill_Latitude[r]), (d1.St_Longitude[r], d1.St_Latitude[r])])
            lc = mc.LineCollection(new_list, colors=f'C{ust + 1}',
                                   alpha=.4,
                                   linewidths=3)  # 'Resource Type' alpha = (ust/len(unique_stations)), colors=ust,
            ax.add_collection(lc)

        # Normalizing the size of circle (oil spill) in the graph so that even if spill size is 80000,
        # circle will be at max size of 500
        x_max = max(spill_df[spill_df['Spill #'].isin([item[0] for item in deploy_1s.index])]['Resource needed'])
        x_min = min(spill_df[spill_df['Spill #'].isin([item[0] for item in deploy_1s.index])]['Resource needed'])

        # Facecolor of ALL spills
        spillAll = plt.scatter(data=spill_df, x='Spill_Longitude', y='Spill_Latitude',
                               s=((spill_df['Resource needed'])
                                  - x_min) * 500 / (x_max - x_min),
                               c=(spill_df['Sensitivity']),
                               cmap='binary')

        # Points of covered spills
        spillC = plt.scatter(data=spill_df[spill_df['Spill #'].isin([item[0] for item in deploy_1s.index])],
                             x='Spill_Longitude', y='Spill_Latitude',
                             s=((spill_df[spill_df['Spill #'].isin([item[0] for item in deploy_1s.index])][
                                 'Resource needed'])
                                - x_min) * 500 / (x_max - x_min),
                             # c=(spill_df[spill_df['Spill #'].isin([item[0] for item in deploy_1s.index])]['Sensitivity']),
                             # cmap='viridis',
                             facecolors='none', edgecolors='green', alpha=0.4)

        # if there is any uncover spill, run the below conditional code block
        if len(spill_df[~spill_df['Spill #'].isin([item[0] for item in deploy_1s.index])]['Resource needed']) > 0:

            # Edge of un-covered spills
            x1_max = max(spill_df[~spill_df['Spill #'].isin([item[0] for item in deploy_1s.index])]['Resource needed'])
            x1_min = min(spill_df[~spill_df['Spill #'].isin([item[0] for item in deploy_1s.index])]['Resource needed'])
            spillUnC = plt.scatter(data=spill_df[~spill_df['Spill #'].isin([item[0] for item in deploy_1s.index])],
                                   x='Spill_Longitude', y='Spill_Latitude',
                                   s=((spill_df[~spill_df['Spill #'].isin([item[0] for item in deploy_1s.index])][
                                       'Resource needed'])
                                      - x1_min) * 500 / (x1_max - x1_min),
                                   # c=(spill_df[~spill_df['Spill #'].isin([item[0] for item in deploy_1s.index])]['Sensitivity']),
                                   # cmap='binary',
                                   facecolors='none', edgecolors='red', alpha=0.5)
        else:
            spillUnC = 0

        # Square showing selected stations
        selected_supply_stations = list(deploy_1s.reset_index().level_1.unique())
        st = plt.scatter(data=station_df[station_df['Station no.'].isin(selected_supply_stations)],
                         x='St_Longitude', y='St_Latitude',
                         marker=shape, alpha=alpha_val, s=facility_size,
                         edgecolors='black',
                         zorder=2,
                         c=station_color)

        # Showing station number as text
        data_st_selected = station_df[station_df['Station no.'].isin(
            select_1s.reset_index().rename({'index': 'StationNo'}, axis=1).StationNo)].reset_index()
        #

        # Showing station number as text for current stations
        if current_vs_proposed == 'current':
            # s_tu, s_ha
            for i in range(len(selected_supply_stations)):
                plt.text(data_st_selected.St_Longitude[i] - 2, y=data_st_selected.St_Latitude[i] - legend_y_position,
                         # -3
                         s=st_name[i], fontdict=dict(color='white', size=12))
        else:
            for i in range(len(selected_supply_stations)):
                plt.text(x=data_st_selected.St_Longitude[i] - 1.1,
                         y=data_st_selected.St_Latitude[i] - legend_y_position,
                         # + 2.5 - .25++
                         s=data_st_selected.loc[:, 'Station no.'][i],
                         zorder=4, fontdict=dict(color=facility_text_color, size=10))
        # Small purple squares to show non-selected stations
        stUns = plt.scatter(data=station_df[~station_df['Station no.'].isin(select_1s.index.tolist())],
                            x='St_Longitude', y='St_Latitude', marker='s', alpha=.25, c='blue')
        # legends of all shapes in this figure
        legend_handle = plt.legend((spillC, spillUnC, st, stUns),
                                   ('Oil Spill covered', 'Oil Spill not covered', 'Station selected',
                                    'Station not selected'),
                                   loc='lower left', ncol=1, handlelength=5, borderpad=.1, markerscale=.4,
                                   fontsize=12)
        """loc='lower left',
                   ncol=1, handlelength=5, borderpad=.05, markerscale=.4,
                   fontsize=14,
                   # handletextpad=0.1              
        """
        legend_handle.legendHandles[0]._sizes = [30]  # same size marker in legend box
        legend_handle.legendHandles[1]._sizes = [30]
        legend_handle.legendHandles[2]._sizes = [30]
        legend_handle.legendHandles[3]._sizes = [30]

        # plt.xticks([])
        # plt.yticks([])
        ax.set_xlim([-141, -60])
        ax.set_ylim([51, 84])

        plt.tight_layout()
        plt.axis('off')
        plt.show()  # ++
        # print(f'\nDistance_max {DistanceMax}, NumberSt_max {NumberStMax}, '
        #      f'num_spills {len(spill_df)}'
        #      f'\nOutputs: Coverage {coverage_percentage}% ResponseTimeT {ResponseTimeT}')
        date_time = str(date.today().strftime("%b %d,") + datetime.now().strftime("%H%M"))

        fig.savefig(
            f'../plots/{current_vs_proposed} ({date_time}) {len(spill_df)} spills {NumberStMax} NumberSt_max {DistanceMax}Distance_max {coverage_percentage}%coverage.png'
            , transparent=False, dpi=500)  # , bbox_inches='tight'

        return assignment_line
