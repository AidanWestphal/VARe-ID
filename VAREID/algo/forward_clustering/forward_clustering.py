#!/usr/bin/env python3
"""
Forward Clustering Algorithm

For each encounter_id group:
- Find the representative annotation (representative=True) and its inter_cluster_id
- Assign that same inter_cluster_id to all non-representative annotations (representative=False)
  in the same encounter_id group
"""

import argparse
import logging
import sys
from pathlib import Path
import uuid

from VAREID.libraries.io.format_funcs import load_config, load_json, save_json
from VAREID.libraries.utils import path_from_file

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def assign_inter_cluster_ids(annotations, cluster_field='encounter_id',
                            inter_field='cluster_id', rep_field='representative'):
    """
    Forward cluster inter_cluster_id from representative to non-representative annotations.

    Logic:
    - For each encounter_id group, find the representative annotation
    - Take the inter_cluster_id from the representative
    - Assign that same inter_cluster_id to all non-representative annotations in the group

    Args:
        annotations: List of annotation dictionaries
        cluster_field: Name of the field containing intra cluster IDs
        inter_field: Name of the field containing inter cluster IDs
        rep_field: Name of the field indicating representative status

    Returns:
        List of annotations with inter_cluster_id propagated from representatives
    """
    # Group annotations by encounter_id
    clusters = {}
    representatives = {}

    for ann in annotations:
        cluster_id = ann.get(cluster_field)

        # Skip annotations without cluster ID
        if cluster_id is None:
            continue

        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(ann)

        # Track representative for each cluster
        if ann.get(rep_field, False):
            if cluster_id in representatives:
                logger.warning(f"Multiple representatives found in cluster {cluster_id}")
            representatives[cluster_id] = ann

    logger.info(f"Found {len(clusters)} unique intra-clusters")
    logger.info(f"Found {len(representatives)} representative annotations")

    # Forward inter_cluster_ids from representatives to non-representatives
    total_assigned = 0
    rep_count = 0
    forward_count = 0
    skipped_no_rep = 0
    skipped_no_inter = 0

    for cluster_id, cluster_annotations in clusters.items():
        # Find the representative annotation for this cluster
        representative = representatives.get(cluster_id)

        if representative is None:
            # No representative in this cluster - shouldn't happen if data is correct
            logger.warning(f"No representative found for intra_cluster {cluster_id} with {len(cluster_annotations)} annotations")
            skipped_no_rep += len(cluster_annotations)
            continue

        # Get the inter_cluster_id from the representative
        representative_inter_id = representative.get(inter_field)

        if representative_inter_id is None:
            logger.warning(f"Representative in intra_cluster {cluster_id} has no cluster_id")
            skipped_no_inter += len(cluster_annotations)
            continue

        # Assign the representative's inter_cluster_id to all annotations in this cluster
        for ann in cluster_annotations:
            # Only update non-representative annotations
            # Representatives should already have their inter_cluster_id
            if not ann.get(rep_field, False):
                ann[inter_field] = representative_inter_id
                forward_count += 1
                total_assigned += 1
            else:
                # Representative already has the ID, just count it
                rep_count += 1
                total_assigned += 1

    # Count annotations without cluster IDs
    null_count = 0
    for ann in annotations:
        if ann.get(cluster_field) is None:
            null_count += 1

    logger.info(f"Forward clustering results:")
    logger.info(f"  - Total annotations processed: {len(annotations)}")
    logger.info(f"  - Representatives (already had cluster_id): {rep_count}")
    logger.info(f"  - Forward clustered (assigned from representative): {forward_count}")
    logger.info(f"  - No encounter_id (skipped): {null_count}")
    logger.info(f"  - Skipped (no representative): {skipped_no_rep}")
    logger.info(f"  - Skipped (representative has no inter_cluster_id): {skipped_no_inter}")

    return annotations


def validate_data(annotations, cluster_field, rep_field):
    """
    Validate that the data has expected structure.

    Args:
        annotations: List of annotation dictionaries
        cluster_field: Name of the field containing intra cluster IDs
        rep_field: Name of the field indicating representative status

    Returns:
        bool: True if validation passes, False otherwise
    """
    # Check that each cluster has exactly one representative
    clusters_reps = {}

    for ann in annotations:
        cluster_id = ann.get(cluster_field)
        if cluster_id is None:
            continue

        is_rep = ann.get(rep_field, False)
        if is_rep:
            if cluster_id not in clusters_reps:
                clusters_reps[cluster_id] = 0
            clusters_reps[cluster_id] += 1

    # Report any issues
    issues = []
    for cluster_id, rep_count in clusters_reps.items():
        if rep_count == 0:
            issues.append(f"Cluster {cluster_id} has no representative")
        elif rep_count > 1:
            issues.append(f"Cluster {cluster_id} has {rep_count} representatives (should be 1)")

    if issues:
        logger.warning("Data validation issues found:")
        for issue in issues[:10]:  # Show first 10 issues
            logger.warning(f"  - {issue}")
        if len(issues) > 10:
            logger.warning(f"  ... and {len(issues) - 10} more issues")
        return False

    return True


def process_annotations_file(input_path, output_path, cluster_field='encounter_id',
                            inter_field='cluster_id', rep_field='representative'):
    """
    Process an annotations file to assign inter cluster IDs.

    Args:
        input_path: Path to input JSON file with annotations
        output_path: Path to save output JSON file
        cluster_field: Name of the field containing intra cluster IDs
        inter_field: Name of the field to store inter cluster IDs
        rep_field: Name of the field indicating representative status
    """
    logger.info(f"Loading annotations from: {input_path}")

    # Load the JSON file
    data = load_json(input_path)

    # Handle both formats: direct list or dict with 'annotations' key
    if isinstance(data, list):
        annotations = data
        output_data = None  # Will save as list
    elif isinstance(data, dict) and 'annotations' in data:
        annotations = data['annotations']
        output_data = data.copy()  # Preserve other fields
    else:
        logger.error(f"Unexpected data format in {input_path}")
        raise ValueError("Input file must be a list of annotations or dict with 'annotations' key")

    logger.info(f"Processing {len(annotations)} annotations")
    logger.info(f"Using intra cluster field: '{cluster_field}'")
    logger.info(f"Using representative field: '{rep_field}'")
    logger.info(f"Assigning to inter cluster field: '{inter_field}'")

    # Check if fields exist
    if annotations:
        sample_ann = annotations[0]
        if cluster_field not in sample_ann:
            logger.warning(f"Cluster field '{cluster_field}' not found in annotations. Available fields: {list(sample_ann.keys())}")
        if rep_field not in sample_ann:
            logger.warning(f"Representative field '{rep_field}' not found in annotations. Available fields: {list(sample_ann.keys())}")

    # Validate data structure
    logger.info("Validating data structure...")
    is_valid = validate_data(annotations, cluster_field, rep_field)
    if not is_valid:
        logger.warning("Data validation found issues, but continuing with processing...")

    # Assign inter cluster IDs
    updated_annotations = assign_inter_cluster_ids(annotations, cluster_field, inter_field, rep_field)

    # Prepare output data
    if output_data is None:
        # Original was a list, save as list
        output_data = updated_annotations
    else:
        # Original was a dict, update annotations
        output_data['annotations'] = updated_annotations

    # Save results
    logger.info(f"Saving results to: {output_path}")
    save_json(output_data, output_path)

    # Calculate and report statistics
    inter_clusters = {}
    intra_clusters = {}
    rep_annotations = 0
    forward_annotations = 0
    null_intra = 0
    null_inter = 0

    for ann in updated_annotations:
        intra_id = ann.get(cluster_field)
        inter_id = ann.get(inter_field)

        # Track intra clusters
        if intra_id is None:
            null_intra += 1
        else:
            if intra_id not in intra_clusters:
                intra_clusters[intra_id] = {'rep': 0, 'forward': 0, 'inter_id': None}

            if ann.get(rep_field, False):
                intra_clusters[intra_id]['rep'] += 1
                intra_clusters[intra_id]['inter_id'] = inter_id
                rep_annotations += 1
            else:
                intra_clusters[intra_id]['forward'] += 1
                forward_annotations += 1

        # Track inter clusters
        if inter_id is None:
            null_inter += 1
        else:
            if inter_id not in inter_clusters:
                inter_clusters[inter_id] = 0
            inter_clusters[inter_id] += 1

    # Calculate average intra cluster size
    intra_cluster_sizes = [c['rep'] + c['forward'] for c in intra_clusters.values()]
    avg_intra_cluster_size = sum(intra_cluster_sizes) / len(intra_cluster_sizes) if intra_cluster_sizes else 0

    logger.info("=" * 60)
    logger.info("FORWARD CLUSTERING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total annotations:                 {len(updated_annotations)}")
    logger.info(f"Unique intra clusters:             {len(intra_clusters)}")
    logger.info(f"Unique inter clusters:             {len(inter_clusters)}")
    logger.info(f"Representative annotations:        {rep_annotations}")
    logger.info(f"Forward clustered:                 {forward_annotations}")
    logger.info(f"No encounter_id (null):       {null_intra}")
    logger.info(f"No cluster_id (null):       {null_inter}")
    logger.info(f"Average intra cluster size:        {avg_intra_cluster_size:.1f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forward cluster_id from representative to non-representative annotations")
    parser.add_argument("input_file", type=str, help="Path to input annotation JSON file")
    parser.add_argument("output_file", type=str, help="Path to output annotation JSON file")
    parser.add_argument("--config", type=str, help="Path to configuration file",
                       default=None)

    args = parser.parse_args()

    # Load configuration
    if args.config and Path(args.config).exists():
        config = load_config(args.config)
    else:
        # Use default config if not provided
        config_path = path_from_file(__file__, "forward_clustering_config.yaml")
        if Path(config_path).exists():
            config = load_config(config_path)
        else:
            # Default parameters
            config = {
                'cluster_field': 'encounter_id',
                'inter_field': 'cluster_id',
                'rep_field': 'representative'
            }
            print("Using default configuration parameters")

    # Validate input file exists
    if not Path(args.input_file).exists():
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)

    # Create output directory if needed
    output_dir = Path(args.output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        process_annotations_file(
            args.input_file,
            args.output_file,
            cluster_field=config.get('cluster_field', 'encounter_id'),
            inter_field=config.get('inter_field', 'cluster_id'),
            rep_field=config.get('rep_field', 'representative')
        )
    except Exception as e:
        logger.error(f"Error processing annotations: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)