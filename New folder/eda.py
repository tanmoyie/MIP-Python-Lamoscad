"""

"""

#%%
# Import Python libraries
# import sweetviz as sv
# from pandas_profiling import ProfileReport
import pandas as pd
import matplotlib.pyplot as plt
import folium
import geopandas as gpd
import shapely
import custom_func
import plotly.express as px
import numpy as np
import matplotlib

#%% Shapefile to Transparent fig
def plot_shp_to_transparent_fig(file_url, name):
    import geopandas as gpd
    fig, ax = plt.subplots(figsize=(8,7))  #++ figsize=(8,8)
    # plt.figure()
    # Load geometric file for map
    Map_shp = gpd.read_file(file_url)
    Map_shp = Map_shp.to_crs(epsg=4326)  # 3857
    Map_Plot = Map_shp.plot(ax=ax,  alpha=.5) # color="lightskyblue",
    ax.set_xlim([-140,-60])
    ax.set_ylim([50, 80])
    ax.grid(False)
    ax.axis('off')
    fig.savefig(f'Outputs/{name}.png', transparent=True)
    # plt.show()


#%%
def plot_point_to_transparent_fig(dataframe, name):
    fig, ax = plt.subplots(figsize=(8,7))
    coordinates = custom_func.extract_coordinate(dataframe)
    coordinate_df = pd.DataFrame(coordinates[0])
    coordinate_df.columns = ['Latitude', 'Longitude']
    coordinate_gdf = gpd.GeoDataFrame(
        coordinate_df, geometry=gpd.points_from_xy(coordinate_df.Longitude, coordinate_df.Latitude))
    coordinate_gdf = coordinate_gdf.set_crs(epsg=4326)

    plt.scatter(data=coordinate_gdf,
                     x='Longitude', y='Latitude', marker='s', alpha=1, s=250,
                     # s=amountSt_groupby['amountSt_display'],
                     c='yellow')
    ax.set_xlim([-140, -60])
    ax.set_ylim([50, 80])
    fig.savefig(f'Outputs/{name}.png', transparent=True)

#%%
# reading geojson file
shipping_route_df = gpd.read_file('Inputs/ArcGIS_data/Shipping_and_Hydrography.geojson')
# shipping_route_df.plot()

#%%
geom = shipping_route_df.geometry
linestring = geom.explode()
# linestring11 = linestring2[11]
# write a for loop for go all LineString
#%%
Hudson = []
for i in range(len(linestring)):
    list1 = list(linestring.geometry.iloc[i].coords)
    H1 = []
    for j in list1:
        if j[0] >= -100 and j[1] < 70:
            H1.append(j)
    Hudson.append(H1)
# Drop empty elements from Hudson list
Hudson = [i for i in Hudson if i]

#%%
Hudson_df = pd.DataFrame()
for k in range(len(Hudson)):
    Hudson_df = pd.concat([Hudson_df, pd.DataFrame(Hudson[k])])
#%%
for j in list1:
    if j[0] >= -100:
        print(j)

#%%
# Produce map
map_st = folium.Map(location=[67, -98], zoom_start=4, min_zoom=2.5, max_zoom=7)
shipping_route = map_st.choropleth(geo_data="Inputs/ArcGIS_data/Shipping_and_Hydrography.geojson")
map_st.save('Outputs/map_st_SN_route.html')

#%% Google Map
def plot_googleMap():
    import gmaps
    gmaps.configure(api_key='AIzaSyAyRAMSfG6ic9oYV9bemRkn2sYqOh8VQSE')
    from bokeh.io import output_notebook
    from bokeh.io import show
    from bokeh.plotting import figure, gmap
    from bokeh.models import GMapOptions

    bokeh_width, bokeh_height =300, 200
    api_key = 'AIzaSyAyRAMSfG6ic9oYV9bemRkn2sYqOh8VQSE'  # os.environ['AIzaSyAyRAMSfG6ic9oYV9bemRkn2sYqOh8VQSE']
    # %%
    output_notebook()
    figure(height=300, width=600, x_range=(-135,-60), y_range=(50, 80))
    def plot(lat, lng, zoom=10, map_type='terrain'):
        gmap_options = GMapOptions(lat=lat, lng=lng,
                                   map_type=map_type, zoom=zoom)
        p = gmap(api_key, gmap_options, title='Google Map',
                 width=bokeh_width, height=bokeh_height)
        show(p)
        return p

    lat, lon = 46.2437, 6.0251
    p = plot(lat, lon)