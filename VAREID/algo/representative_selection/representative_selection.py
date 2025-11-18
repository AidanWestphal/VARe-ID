#!/usr/bin/env python3
"""
Representative Selection Algorithm

For each group of annotations with the same intra_cluster_id,
selects the annotation with the highest IA score as the representative.
All other annotations in the group are marked as non-representative.
"""

import argparse
import logging
import sys
from pathlib import Path

from VAREID.libraries.io.format_funcs import load_json, save_json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def select_representatives(annotations, cluster_field='intra_cluster_id', ia_field='ia_score'):
    """
    Select representative annotations based on highest IA score within each cluster.

    Args:
        annotations: List of annotation dictionaries
        cluster_field: Name of the field containing cluster IDs
        ia_field: Name of the field containing IA scores

    Returns:
        List of annotations with 'representative' field added
    """
    # Group annotations by cluster ID
    clusters = {}
    for idx, ann in enumerate(annotations):
        cluster_id = ann.get(cluster_field)

        # Skip annotations without cluster ID
        if cluster_id is None:
            ann['representative'] = False
            continue

        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append((idx, ann))

    logger.info(f"Found {len(clusters)} unique clusters")

    # For each cluster, find the annotation with highest IA score
    representative_count = 0
    for cluster_id, cluster_annotations in clusters.items():
        if len(cluster_annotations) == 0:
            continue

        # Find annotation with highest IA score
        best_idx = None
        best_score = -1

        for idx, ann in cluster_annotations:
            ia_score = ann.get(ia_field, 0)

            # Convert to float if necessary
            try:
                ia_score = float(ia_score)
            except (TypeError, ValueError):
                ia_score = 0

            if ia_score > best_score:
                best_score = ia_score
                best_idx = idx

        # Mark annotations as representative or not
        for idx, ann in cluster_annotations:
            if idx == best_idx:
                ann['representative'] = True
                representative_count += 1
            else:
                ann['representative'] = False

    logger.info(f"Selected {representative_count} representative annotations")

    # Mark any annotations without cluster IDs as non-representative
    for ann in annotations:
        if 'representative' not in ann:
            ann['representative'] = False

    return annotations


def process_annotations_file(input_path, output_path, cluster_field='intra_cluster_id', ia_field='ia_score'):
    """
    Process an annotations file to select representatives.

    Args:
        input_path: Path to input JSON file with annotations
        output_path: Path to save output JSON file
        cluster_field: Name of the field containing cluster IDs
        ia_field: Name of the field containing IA scores
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
    logger.info(f"Using cluster field: '{cluster_field}'")
    logger.info(f"Using IA score field: '{ia_field}'")

    # Check if fields exist
    if annotations:
        sample_ann = annotations[0]
        if cluster_field not in sample_ann:
            logger.warning(f"Cluster field '{cluster_field}' not found in annotations. Available fields: {list(sample_ann.keys())}")
        if ia_field not in sample_ann:
            logger.warning(f"IA score field '{ia_field}' not found in annotations. Available fields: {list(sample_ann.keys())}")

    # Select representatives
    updated_annotations = select_representatives(annotations, cluster_field, ia_field)

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
    total_annotations = len(updated_annotations)
    representatives = sum(1 for ann in updated_annotations if ann.get('representative', False))
    non_representatives = total_annotations - representatives

    logger.info("=" * 60)
    logger.info("REPRESENTATIVE SELECTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total annotations:     {total_annotations}")
    logger.info(f"Representatives:       {representatives} ({representatives/total_annotations*100:.1f}%)")
    logger.info(f"Non-representatives:   {non_representatives} ({non_representatives/total_annotations*100:.1f}%)")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Select representative annotations based on highest IA score within each cluster"
    )

    parser.add_argument(
        "input_path",
        type=str,
        help="Path to input JSON file with annotations containing cluster IDs and IA scores"
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to save output JSON file with representative field added"
    )
    parser.add_argument(
        "--ia_field",
        type=str,
        default="ia_score",
        help="Name of the field containing IA scores (default: ia_score)"
    )
    parser.add_argument(
        "--cluster_field",
        type=str,
        default="intra_cluster_id",
        help="Name of the field containing cluster IDs (default: intra_cluster_id)"
    )

    args = parser.parse_args()

    # Validate input file exists
    if not Path(args.input_path).exists():
        logger.error(f"Input file not found: {args.input_path}")
        sys.exit(1)

    # Create output directory if needed
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        process_annotations_file(
            args.input_path,
            args.output_path,
            cluster_field=args.cluster_field,
            ia_field=args.ia_field
        )
    except Exception as e:
        logger.error(f"Error processing annotations: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()