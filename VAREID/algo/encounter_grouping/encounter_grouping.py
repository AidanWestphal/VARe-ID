
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from geopy.distance import geodesic
import datetime

from VAREID.libraries.io.format_funcs import load_config, load_json, save_json, join_dataframe_dict, split_dataframe
from VAREID.libraries.utils import path_from_file


def calculate_distance_matrix(locations):
    """
    Calculate pairwise distance matrix between GPS coordinates in kilometers.
    
    Args:
        locations: List of (lat, lon) tuples
    
    Returns:
        Distance matrix as numpy array
    """
    n = len(locations)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            try:
                dist = geodesic(locations[i], locations[j]).kilometers
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
            except:
                # Handle invalid coordinates by setting large distance
                dist_matrix[i, j] = 999999
                dist_matrix[j, i] = 999999
    
    return dist_matrix


def calculate_time_distance_matrix(times):
    """
    Calculate pairwise time difference matrix in hours.
    
    Args:
        times: List of timestamps (POSIX or other format)
    
    Returns:
        Time difference matrix as numpy array
    """
    times_array = np.array(times)
    time_matrix = np.abs(times_array[:, np.newaxis] - times_array)
    # Convert seconds to hours
    return time_matrix / 3600.0


def group_encounters(images_df, config):
    """
    Group images into encounters based on GPS location and time proximity.
    Images are grouped if they are within BOTH distance AND time thresholds.
    
    Args:
        images_df: DataFrame containing image metadata
        config: Configuration parameters
    
    Returns:
        DataFrame with added encounter_id column
    """
    # Filter out images without GPS or time data
    valid_images = images_df.dropna(subset=['gps_lat', 'gps_lon', 'timestamp']).copy()
    
    if len(valid_images) == 0:
        print("No images with valid GPS and time data found")
        images_df['encounter_id'] = -1
        return images_df
    
    # Extract coordinates and times
    locations = list(zip(valid_images['gps_lat'], valid_images['gps_lon']))
    times = valid_images['timestamp'].tolist()
    
    print(f"Processing {len(locations)} images with valid GPS/time data")
    
    # Calculate distance matrices
    spatial_distances = calculate_distance_matrix(locations)
    temporal_distances = calculate_time_distance_matrix(times)
    
    # Get thresholds from config
    max_distance_km = config['max_distance_km']
    max_time_hours = config['max_time_hours']
    
    # Create binary connectivity matrix: 1 if within both thresholds, 0 otherwise
    n = len(locations)
    connectivity = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                connectivity[i, j] = 0  # Distance to self is 0 for DBSCAN
            elif (spatial_distances[i, j] <= max_distance_km and 
                  temporal_distances[i, j] <= max_time_hours):
                connectivity[i, j] = 0.5  # Within both thresholds
            else:
                connectivity[i, j] = 2.0  # Outside at least one threshold
    
    # Apply DBSCAN clustering
    clustering = DBSCAN(
        eps=1.0,  # Threshold between connected (0.5) and disconnected (2.0)
        min_samples=config.get('min_samples_per_encounter', 1),
        metric='precomputed'
    )
    
    cluster_labels = clustering.fit_predict(connectivity)
    
    # Add encounter IDs to valid images
    valid_images['encounter_id'] = cluster_labels
    
    # Handle outliers (label -1) by assigning unique encounter IDs
    max_encounter_id = cluster_labels.max() if cluster_labels.max() >= 0 else -1
    outlier_mask = cluster_labels == -1
    n_outliers = outlier_mask.sum()
    if n_outliers > 0:
        outlier_ids = range(max_encounter_id + 1, max_encounter_id + 1 + n_outliers)
        valid_images.loc[outlier_mask, 'encounter_id'] = outlier_ids
    
    # Merge back with original images DataFrame
    images_df = images_df.merge(
        valid_images[['image_uuid', 'encounter_id']], 
        on='image_uuid', 
        how='left'
    )
    
    # Fill missing encounter IDs for images without GPS/time data
    mask = images_df['encounter_id'].isna()
    current_max = pd.to_numeric(images_df['encounter_id'], errors='coerce').max()
    start_id = (int(current_max) if pd.notna(current_max) else -1) + 1
    images_df.loc[mask, 'encounter_id'] = np.arange(start_id, start_id + mask.sum(), dtype=int)
  # -2 indicates no GPS/time data
    
    print(f"Created {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)} main encounters")
    print(f"Found {n_outliers} outlier images (single-image encounters)")
    print(f"Thresholds used: {max_distance_km}km, {max_time_hours}h")
    
    return images_df


def save_encounter_results(input_path, output_path, config):
    """
    Load annotation data, group into encounters, and save results.
    
    Args:
        input_path: Path to input annotation JSON file
        output_path: Path to save output annotation JSON file
        config: Configuration parameters
    """
    print(f"Loading annotations from: {input_path}")
    
    # Load and join annotation data
    data = join_dataframe_dict(load_json(input_path))
    
    # Convert images to DataFrame for processing
    images_df = pd.DataFrame(data['images'])
    
    # Group images into encounters
    images_with_encounters = group_encounters(images_df, config)
    
    # Update annotations with encounter information
    annotations_df = pd.DataFrame(data['annotations'])
    
    # Merge encounter IDs into annotations based on image_uuid
    if 'image_uuid' in annotations_df.columns:
        encounter_mapping = images_with_encounters.set_index('image_uuid')['encounter_id'].to_dict()
        annotations_df['encounter_id'] = annotations_df['image_uuid'].map(encounter_mapping)
    else:
        print("Warning: No image_uuid found in annotations")
        annotations_df['encounter_id'] = -2
    
    # Update the data structure
    data['images'] = images_with_encounters.to_dict('records')
    data['annotations'] = annotations_df.to_dict('records')
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_json(data, output_path)
    
    print(f"Saved encounter-grouped annotations to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Group images into encounters based on GPS and time proximity")
    parser.add_argument("input_file", type=str, help="Path to input annotation JSON file")
    parser.add_argument("output_file", type=str, help="Path to output annotation JSON file")
    parser.add_argument("--config", type=str, help="Path to configuration file", 
                       default=None)
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
    else:
        # Use default config if not provided
        config_path = path_from_file(__file__, "encounter_grouping_config.yaml")
        if os.path.exists(config_path):
            config = load_config(config_path)
        else:
            # Default parameters
            config = {
                'max_distance_km': 1.0,
                'max_time_hours': 24.0,
                'min_samples_per_encounter': 1
            }
            print("Using default configuration parameters")
    
    save_encounter_results(args.input_file, args.output_file, config)
