#!/usr/bin/env python
# coding: utf-8

# Let's first import some libraries
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
import h3
import folium
from shapely.geometry import Polygon, Point, mapping, LineString
from libpysal import weights
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from feature_engine.wrappers import SklearnTransformerWrapper
from geopandas import GeoDataFrame
from sklearn.preprocessing import MinMaxScaler
from geopy.distance import geodesic
import shapely

def check_dependencies():
    try:
        import pandas
        import numpy
    except ImportError:
        print("Please install the required dependencies by running 'pip install -r requirements.txt'")
        exit(1)

def main():
    check_dependencies()

    parser = argparse.ArgumentParser(description='Assign a starting point to a user and get potential group members.')
    parser.add_argument('--input_file', required = True, help='Path to the input csv file')

    args = parser.parse_args()

    # Pass in the input csv file and read it in
    df = pd.read_csv(args.input_file)


    # Let's first create a columns called number_of_races to state how many rides that rider has, so we can 
    # analyze them
    df['number_of_rides'] = df.groupby('user_id')['user_id'].transform('count')

    # Let's create a column called same_starting_point that's a binary flag: 0 if it's a different starting point
    # and 1 if it's the same starting point
    mask = df.groupby('user_id')[['latitude', 'longitude']].apply(lambda x: x.nunique().eq(1)).all(axis=1)
    df['same_starting_point'] = df['user_id'].isin(mask[mask].index).astype(int)

    # For cyclists that have more than one ride, let's see if it's the same starting point. If not, how far
    # they are between each other
    df[(df['number_of_rides'] > 1) & (df['same_starting_point'] == 0)]['user_id'].value_counts()

    ### Let's see how many cyclists have the same starting point and how many have different starting points
    df.groupby('same_starting_point').user_id.nunique()

    # Let's create a column called distance_between_rides to denote how far from each other the starting points
    # are for people with multiple rides
    # First, let's create a coordinate out of the latitude and longitude, because we will need this to
    # take the difference between two coordinates in km
    df['coordinate_point'] = list(zip(df['latitude'], df['longitude']))

    # Next, let's create a column called next_ride_coordinate so we can take the difference for each user
    df = df.sort_values(by=['user_id'])

    # Create a new column 'next_ride_coordinate' with the next row's 'coordinate_point' for each user
    df['next_ride_coordinate'] = df.groupby('user_id')['coordinate_point'].shift(-1)

    # Fill the 'next_ride_coordinate' for the last row of each user with the last coordinate point
    df['next_ride_coordinate'] = df.groupby('user_id')['next_ride_coordinate'].ffill()

    # For users with only one ride, assign the location_tuple to the next_ride_coordinate
    single_ride_users = df['user_id'][df['number_of_rides'] == 1].unique()
    df.loc[df['user_id'].isin(single_ride_users), 'next_ride_coordinate'] = df['coordinate_point']

    # Finally, let's create the distance_between_rides column
    df['distance_between_rides'] = df.apply(lambda row: geodesic(row['coordinate_point'], row['next_ride_coordinate']).kilometers, axis=1)

    # Let's create a new column called central_location for users that have multiple rides. The assumption is,
    # maybe the cyclist starts from this central location and goes to a starting point to cycle

    # Function to calculate the centroid for a set of coordinates
    def calculate_centroid(locations):
        num_points = len(locations)
        
        if num_points == 0:
            return None
        
        avg_latitude = sum(lat for lat, lon in locations) / num_points
        avg_longitude = sum(lon for lat, lon in locations) / num_points
        
        return avg_latitude, avg_longitude

    # Group by user_id and apply the calculate_centroid function to each user's location tuples
    centroid = df.groupby('user_id')['coordinate_point'].apply(calculate_centroid)

    # Merge the calculated centroids back to the original DataFrame
    df = df.merge(centroid.rename('central_location'), left_on='user_id', right_index=True)

    # Unpack centroid tuples into separate latitude and longitude columns
    df['centroid_latitude'], df['centroid_longitude'] = zip(*df['central_location'])

    df = df.drop_duplicates(subset=['user_id', 'centroid_latitude', 'centroid_longitude'])

    df = df.drop(columns=['latitude', 'longitude', 'number_of_rides', 'same_starting_point', 'coordinate_point',
                           'next_ride_coordinate', 'distance_between_rides'])

    geometry = [Point(xy) for xy in df['central_location']]
    gdf = GeoDataFrame(df, crs="EPSG:4326", geometry=geometry)

    def get_hexagon_grid(latitude, longitude, resolution, ring_size):

        # Get the H3 hexagons covering the specified location
        center_h3 = h3.geo_to_h3(latitude, longitude, resolution)
        hexagons = list(h3.k_ring(center_h3, ring_size))  

        # Create a GeoDataFrame with hexagons and their corresponding geometries
        hexagon_geometries = [shapely.geometry.Polygon(h3.h3_to_geo_boundary(hexagon, geo_json=True)) for hexagon in hexagons]
        hexagon_df = gpd.GeoDataFrame({'hexagon_id': hexagons, 'geometry': hexagon_geometries})

        return hexagon_df

    # Latitude and longitude coordinates for the center of the area
    center_lat = gdf['centroid_latitude'].mean()
    center_lng = gdf['centroid_longitude'].mean()

    # Generate H3 hexagons at a specified resolution 
    resolution = 5

    # Indicate the number of rings around the central hexagon
    ring_size = 4 # I tried a lot of these before I settled for this number, everything below returned null hexagons for some people
                                         
    # Hexagon grid around the area in the dataset
    hexagon_df = get_hexagon_grid(center_lat, center_lng, resolution, ring_size)

    def calculate_hexagon_ids(df, hexagon_df):

        # Create a column hexagon_id with the id of the hexagon, hexagon_center_latitude, and hexagon_center_longitude
        df['hexagon_id'] = None
        df['hexagon_center_latitude'] = None
        df['hexagon_center_longitude'] = None

        # Iterate through the rides and assign them to a hexagon
        for i, user in df.iterrows():
            point = Point(user["centroid_longitude"], user["centroid_latitude"]) 
            for _, row in hexagon_df.iterrows():
                if point.within(row['geometry']):
                    df.loc[i, 'hexagon_id'] = row['hexagon_id']
                    df.loc[i, 'hexagon_center_latitude'] = row['geometry'].centroid.y
                    df.loc[i, 'hexagon_center_longitude'] = row['geometry'].centroid.x
        
        return df

    result = calculate_hexagon_ids(df, hexagon_df)

    # Convert hexagon column to categorical and get integer codes
    hexagon_categories = pd.Categorical(result['hexagon_id'])
    result['hexagon_id'] = hexagon_categories.codes + 1  # Adding 1 to start indexing from 1

    # Group by 'hexagon_id' and aggregate 'user_id' into a list for each group
    grouped_data = result.groupby('hexagon_id')['user_id'].agg(list).reset_index()

    # Merge the aggregated data back into the original DataFrame
    result = pd.merge(result, grouped_data, on='hexagon_id', how='left', suffixes=('', '_group'))

    # Create the 'potential_group_members' column 
    result['potential_group_members'] = result.apply(lambda row: [user_id for user_id in row['user_id_group'] if user_id != row['user_id']], axis=1)

    # Drop unnecessary columns
    result.drop(['user_id_group', 'central_location', 'centroid_latitude', 'centroid_longitude'], axis=1, inplace=True)

    result = result.rename(columns={'hexagon_id': 'starting_point_id', 'hexagon_center_latitude': 'starting_point_latitude', 'hexagon_centar_longitude': 'starting_point_longitude'})

    output_file = 'result.csv'
    result.to_csv(output_file, index=False)

    print(f"Results have been saved to the output file: {output_file}")
        

if __name__ == '__main__':
    main()





