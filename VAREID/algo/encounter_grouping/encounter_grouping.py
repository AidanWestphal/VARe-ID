
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import datetime

from VAREID.libraries.io.format_funcs import load_config, load_json, save_json, join_dataframe_dict, join_dataframe, split_dataframe
from VAREID.libraries.utils import path_from_file


def calculate_distance_matrix(locations):
    """
    Calculate pairwise distance matrix between GPS coordinates in kilometers.
    Uses vectorized haversine formula for performance.

    Args:
        locations: List of (lat, lon) tuples

    Returns:
        Distance matrix as numpy array
    """
    coords = np.array(locations, dtype=np.float64)
    lat = np.radians(coords[:, 0])
    lon = np.radians(coords[:, 1])

    dlat = lat[:, np.newaxis] - lat[np.newaxis, :]
    dlon = lon[:, np.newaxis] - lon[np.newaxis, :]

    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, np.newaxis]) * np.cos(lat[np.newaxis, :]) * np.sin(dlon / 2) ** 2
    dist_matrix = 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

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


def group_encounters(joined_df, config):
    """
    Group images into encounters based on GPS location and time proximity.
    Images are grouped if they are within BOTH distance AND time thresholds.

    Args:
        joined_df: DataFrame containing joined annotation and image metadata
        config: Configuration parameters

    Returns:
        DataFrame with occurence_id added only to image-level columns
    """
    # Extract unique images with their GPS/time data
    # Since we have multiple annotations per image, get unique images first
    image_cols = ['image_uuid', 'gps_lat', 'gps_lon', 'timestamp']
    available_cols = [col for col in image_cols if col in joined_df.columns]

    if not all(col in joined_df.columns for col in image_cols):
        print(f"Warning: Missing required columns. Available: {available_cols}")
        # If GPS/time data is missing, assign default encounter IDs
        joined_df['occurence_id'] = 0
        return joined_df

    # Get unique images (one row per image_uuid)
    images_df = joined_df[image_cols].drop_duplicates(subset=['image_uuid']).copy()

    # Filter out images without GPS or time data
    valid_images = images_df.dropna(subset=['gps_lat', 'gps_lon', 'timestamp']).copy()

    if len(valid_images) == 0:
        print("No images with valid GPS and time data found")
        # Create a mapping of image_uuid to default occurence_id
        images_df['occurence_id'] = range(len(images_df))
    else:
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
        within_both = (spatial_distances <= max_distance_km) & (temporal_distances <= max_time_hours)
        connectivity = np.where(within_both, 0.5, 2.0)
        np.fill_diagonal(connectivity, 0.0)

        # Apply DBSCAN clustering
        clustering = DBSCAN(
            eps=1.0,  # Threshold between connected (0.5) and disconnected (2.0)
            min_samples=config.get('min_samples_per_encounter', 1),
            metric='precomputed'
        )

        cluster_labels = clustering.fit_predict(connectivity)

        # Add encounter IDs to valid images
        valid_images['occurence_id'] = cluster_labels

        # Handle outliers (label -1) by assigning unique encounter IDs
        max_occurence_id = cluster_labels.max() if cluster_labels.max() >= 0 else -1
        outlier_mask = cluster_labels == -1
        n_outliers = outlier_mask.sum()
        if n_outliers > 0:
            outlier_ids = range(max_occurence_id + 1, max_occurence_id + 1 + n_outliers)
            valid_images.loc[outlier_mask, 'occurence_id'] = list(outlier_ids)

        # Merge back with all images
        images_df = images_df.merge(
            valid_images[['image_uuid', 'occurence_id']],
            on='image_uuid',
            how='left'
        )

        # Fill missing encounter IDs for images without GPS/time data
        mask = images_df['occurence_id'].isna()
        if mask.any():
            current_max = pd.to_numeric(images_df['occurence_id'], errors='coerce').max()
            start_id = (int(current_max) if pd.notna(current_max) else -1) + 1
            images_df.loc[mask, 'occurence_id'] = range(start_id, start_id + mask.sum())

        print(f"Created {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)} main encounters")
        print(f"Found {n_outliers} outlier images (single-image encounters)")
        print(f"Thresholds used: {max_distance_km}km, {max_time_hours}h")

    # Now merge the occurence_id back to the full joined dataframe
    # This ensures each annotation gets the occurence_id of its image
    joined_df = joined_df.merge(
        images_df[['image_uuid', 'occurence_id']],
        on='image_uuid',
        how='left'
    )

    return joined_df


def save_encounter_results(input_path, output_path, config):
    """
    Load annotation data, group into encounters, and save results.
    Uses the standard split/join dataframe logic to preserve original field names.

    Args:
        input_path: Path to input annotation JSON file
        output_path: Path to save output annotation JSON file
        config: Configuration parameters
    """
    print(f"Loading annotations from: {input_path}")

    # Load annotation data and join into a single dataframe
    # This applies reverse renaming to standardize field names
    joined_df = join_dataframe(load_json(input_path))

    # Group images into encounters
    # This adds the occurence_id column to the dataframe
    joined_df_with_encounters = group_encounters(joined_df, config)

    # Split the dataframe back into the standard format
    # This applies forward renaming to restore original field names
    output_data = split_dataframe(joined_df_with_encounters)

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_json(output_data, output_path)

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
