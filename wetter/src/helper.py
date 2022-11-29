from sqlalchemy import create_engine
import geopandas as gpd
import pandas as pd
import numpy as np
from meteostat import Stations, Daily, Point
from datetime import datetime
import matplotlib.pyplot as plt

def get_nuts_data(lvl:int) -> pd.DataFrame:
    data = gpd.read_file(f"./data/nuts5000/5000_NUTS{lvl}.shp")
    # Calculate centroids
    ct = data.to_crs(epsg=3035).centroid.to_crs(epsg=4326)
    X = ct.map(lambda p: p.x) * 100
    Y = ct.map(lambda p: p.y) * 100
    X = X.astype(int)
    Y = Y.astype(int)
    centroids = pd.DataFrame({'lon_times100':X, 'lat_times100':Y})
    nuts = data[['NUTS_LEVEL', 'NUTS_CODE', 'NUTS_NAME']]
    return pd.concat([nuts, centroids], axis=1)

def plot_centroids_with_radius(lvl:int):
    data = gpd.read_file(f"./data/nuts5000/5000_NUTS{lvl}.shp")
    data.geometry = data.geometry.to_crs("EPSG:32634")
    fig, ax = plt.subplots()
    
    # plot germany
    data.plot(color='white', edgecolor='black', ax=ax)

    # plot 50km buffer
    data.centroid.buffer(50000, resolution=6).plot(ax=ax, color='blue', alpha=0.4, cmap='Pastel1')
    
    # plot cerntoids
    data.centroid.plot(color='black', marker='x', markersize=2, label='centroids', ax=ax)
    
    ax.legend()
    plt.show()

def get_daily_weather_data(
    lat:float, lon:float, 
    radius:int, start:datetime, 
    end:datetime
) -> pd.DataFrame:
    # Fetch the stations
    stations = Stations()
    stations = stations.nearby(lat, lon, radius)
    station_data = stations.fetch()
    # Run all the transformations
    daily = Daily(station_data, start, end)
    data = daily.normalize().interpolate().aggregate('1D', spatial=True).fetch()
    # Return data
    return data

def get_daily_weather_data_loc(
    loc: tuple[str, str], 
    start:datetime, 
    end:datetime) -> pd.DataFrame:
    # fetch stations
    stations = Stations()
    stations = stations.region(*loc)
    station_data = stations.fetch()
    # Run all the transformations
    daily = Daily(station_data, start, end)
    data = daily.normalize().interpolate().aggregate('1D', spatial=True).fetch()
    # Return data
    return data