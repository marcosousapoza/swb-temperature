import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm

##########################
# READING DATA FUNCTIONS #
##########################

def get_station_data(dirpath:str, ids:list[str], bar=True) -> pd.DataFrame:

    # the concatonated dataframe variable
    rdf = pd.DataFrame()

    # create progress bar
    if bar:
        pbar = tqdm(ids, leave=False)
    else:
        pbar = ids

    # read files
    for i in pbar:
        filename = f'{dirpath}/{i}.csv'

        # set discription of progress bar
        if bar:
            pbar.set_description(f"Reading {filename}")
        new = pd.read_csv(filename, converters={'id': str})

        # read file
        if rdf.empty:
            rdf = new
        else:
            rdf = pd.concat([rdf, new], ignore_index=True, axis=0)
    
    rdf['time'] = pd.to_datetime(rdf['time'])
    return rdf


def get_nuts_weather_data(infofile:str, dirpath:str, nutslvl:int, code:str=None, bar=True) -> pd.DataFrame:
    
    # read the station info
    stations = gpd.read_file(infofile, converters={'id': str})
    stations['id'] = stations['id'].astype(str)

    # if code is specified only data from that area is selected
    if code:
        stations = stations[stations[f'NUTS_CODE_{nutslvl}'] == code]

    if stations.empty:
        raise ValueError("The parameters you gave do not give any results. Check them")

    # read station data
    ids = stations['id'].tolist()
    data = get_station_data(dirpath, ids, bar=bar)
    data['id'] = data['id'].astype(str)

    # join the dataframes
    joined:pd.DataFrame
    joined = stations.merge(data, on='id', how='right')

    # group by time and nuts level and then take the average
    result = joined.groupby([f'NUTS_CODE_{nutslvl}', 'time']).mean()

    # return the result
    return result


def get_point_weather_data(point:tuple[float, float], max_distance:float|int) -> pd.DataFrame:
    #TODO test this function!!!
    #TODO make function such that a vector of points can be used.

    # read the station info
    stations:gpd.GeoDataFrame
    stations = gpd.read_file('./prod/stationinfo.geojson')

    # calculate distance of each station to point
    stations['distance'] = stations[['geometry']].distance(point)

    # filter stations that are too far away
    stations = stations[stations['distance'] <= max_distance]

    # calculate weight
    stations['weight'] = 1 / stations['distance']

    # get the station data
    data = get_station_data(list(stations['id']))
    data.join(stations, on='id', how='left')
    
    # group all the stations by date
    grouped = data.groupby('time')

    # compute the weighted average
    #create the custom weighted mean aggregation function
    value_cols = 'tavg tmin tmax prcp snow wdir wspd wpgt pres tsun'.split() # values to aggregate
    wm = lambda x, w: np.average(x, weights=w)
    agg_funcs = {col: wm for col in value_cols}
    result = grouped[[*value_cols, 'weight']].aggregate(agg_funcs)

    # return the result
    return result


######################
# PLOTTING FUNCTIONS #
######################

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


"""
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
"""