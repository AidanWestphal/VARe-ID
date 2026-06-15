import argparse
import os
import subprocess
import sys
import shutil
import yaml

import pandas as pd

from VAREID.libraries.io.format_funcs import clone_from_github, load_config, load_json, save_json, split_dataframe, join_dataframe_dict
from VAREID.libraries.utils import path_from_file


def save_lca_results(input_dir, anno_file, output_path, prefix, suffix, field_filters=None, uuid_key="annot_uuid"):
    """
    Save LCA results with support for multiple field filtering.

    Args:
        field_filters: dict of {field_name: field_value} to filter on, e.g. {'viewpoint': 'left', 'encounter': 'E001'}
    """
    clustering_file = os.path.join(input_dir, "clustering.json")
    node2uuid_file = os.path.join(input_dir, "node2uuid_file.json")

    print(f"anno_file: {anno_file}")
    # Load original annotation file
    data = join_dataframe_dict(load_json(anno_file))

    # Load clustering results
    clusters = load_json(clustering_file)
    node2uuid = load_json(node2uuid_file)

    # Build mapping from UUID to cluster ID
    uuid_to_cluster = {}
    for cluster_id, nodes in clusters.items():
        for node in nodes:
            uuid = node2uuid.get(str(node))
            if uuid:
                uuid_to_cluster[uuid] = cluster_id

    # Filter annotations based on field filters (substring matching)
    if field_filters:
        print(field_filters)
        filtered_annotations = []
        for ann in data['annotations']:
            match = True
            for field, value in field_filters.items():
                # Check direct field or name_<field> pattern
                field_value = ann.get(field) or ann.get(f"name_{field}")
                if field_value is None:
                    match = False
                    break
                # Substring matching (consistent with new filtering logic)
                if str(value) not in str(field_value):
                    match = False
                    break
            if match:
                filtered_annotations.append(ann)

        filter_desc = ', '.join(f"{k}={v}" for k, v in field_filters.items())
        print(f"Filtered {len(filtered_annotations)} annotations with {filter_desc} "
              f"out of {len(data['annotations'])}")
    else:
        filtered_annotations = data['annotations']

    # Add LCA_clustering_id to each annotation
    # for ann in filtered_annotations:
    #     ann['LCA_clustering_id'] = uuid_to_cluster.get(ann[uuid_key], None)

    # Build output path
    if field_filters:
        field_str = '_'.join(f"{k}-{v}" for k, v in field_filters.items())
        output_filename = f"{prefix}_{field_str}_{suffix}.json"
    else:
        output_filename = f"{prefix}_{suffix}.json"
    output_path_full = os.path.join(output_path, output_filename)

    # Save final result with same categories/images, modified annotations
    result_dict = split_dataframe(pd.DataFrame(filtered_annotations))

    os.makedirs(os.path.dirname(output_path_full), exist_ok=True)
    save_json(result_dict, output_path_full)
    print(f"Saved LCA results to {output_path_full}")


def save_lca_results_unified(combinations, anno_file, output_path, prefix, suffix, uuid_key="annot_uuid"):
    """
    Save unified LCA results from multiple field combinations into a single file.
    Cluster IDs are remapped to avoid collisions across different field combinations.

    Args:
        combinations: list of (input_dir, field_filters) tuples
        anno_file: path to the original annotation file
        output_path: directory to save the output file
        prefix: output file prefix
        suffix: output file suffix
        uuid_key: key to use for UUID lookups
    """
    print(f"Processing {len(combinations)} field combinations for unified output...")

    # Load original annotation file once
    data = join_dataframe_dict(load_json(anno_file))

    # Track cluster ID offset and collect all annotations with updated cluster IDs
    cluster_offset = 0
    all_uuid_to_cluster = {}

    for input_dir, field_filters in combinations:
        clustering_file = os.path.join(input_dir, "clustering.json")
        node2uuid_file = os.path.join(input_dir, "node2uuid_file.json")

        # Check if clustering files exist
        if not os.path.exists(clustering_file) or not os.path.exists(node2uuid_file):
            filter_desc = ', '.join(f"{k}={v}" for k, v in field_filters.items()) if field_filters else "no filters"
            print(f"Warning: Clustering files not found for {filter_desc}, skipping...")
            continue

        # Load clustering results
        clusters = load_json(clustering_file)
        node2uuid = load_json(node2uuid_file)

        # Build mapping from UUID to cluster ID with offset
        uuid_to_cluster_local = {}
        max_cluster_id = -1

        for cluster_id, nodes in clusters.items():
            # Convert cluster_id to int for remapping (handle string cluster IDs)
            try:
                cluster_id_int = int(cluster_id)
            except (ValueError, TypeError):
                # Handle non-numeric cluster IDs (e.g., noise markers)
                cluster_id_int = -1

            for node in nodes:
                uuid = node2uuid.get(str(node))
                if uuid:
                    # Apply offset to valid cluster IDs
                    if cluster_id_int >= 0:
                        remapped_cluster_id = str(cluster_id_int + cluster_offset)
                        uuid_to_cluster_local[uuid] = remapped_cluster_id
                        max_cluster_id = max(max_cluster_id, cluster_id_int)
                    else:
                        # Keep special cluster IDs as-is (e.g., -1 for noise)
                        uuid_to_cluster_local[uuid] = cluster_id

        # Merge into global mapping
        all_uuid_to_cluster.update(uuid_to_cluster_local)

        # Update offset for next iteration
        if max_cluster_id >= 0:
            cluster_offset += max_cluster_id + 1

        filter_desc = ', '.join(f"{k}={v}" for k, v in field_filters.items()) if field_filters else "no filters"
        print(f"  Processed {len(uuid_to_cluster_local)} UUIDs for {filter_desc}, "
              f"cluster offset now at {cluster_offset}")

    # Add LCA_clustering_id to all annotations that have a cluster assignment
    annotations_with_clusters = 0
    for ann in data['annotations']:
        cluster_id = all_uuid_to_cluster.get(ann[uuid_key])
        if cluster_id is not None:
            ann[output_key] = cluster_id
            annotations_with_clusters += 1
        else:
            ann[output_key] = None

    print(f"Added cluster IDs to {annotations_with_clusters} annotations out of {len(data['annotations'])} total")

    # Build output path
    output_filename = f"{prefix}_{suffix}.json"
    output_path_full = os.path.join(output_path, output_filename)

    # Save final result with categories, images, and modified annotations
    result_dict = {
        'categories': data.get('categories', []),
        'images': data.get('images', []),
        'annotations': data['annotations']
    }

    os.makedirs(os.path.dirname(output_path_full), exist_ok=True)
    save_json(result_dict, output_path_full)
    print(f"Saved unified LCA results to {output_path_full}")
    

if __name__ == "__main__":
    print("Loading data...")
    parser = argparse.ArgumentParser(description="Run LCA clustering")
    parser.add_argument("annots", type=str, help="The path to the annotation file.")
    parser.add_argument("embeddings", type=str, help="The path to the embeddings file.")
    #parser.add_argument("verifiers_probs", type=str, help="The path to the verifier probabilities.")
    parser.add_argument("lca_dir", type=str, help="The directory to save files into.")
    parser.add_argument("log_subunit_file", type=str, help="The path to the log file for the LCA algorithm itself.")
    parser.add_argument("log_file", type=str, help="The path to the log file.")
    parser.add_argument("--video", action="store_true", help="True if LCA should run on the video (drone) config file.")
    parser.add_argument("--intra", action="store_true", help="True if LCA should run on the intra config file.")
    parser.add_argument("--inter", action="store_true", help="True if LCA should run on the inter config file.")
    parser.add_argument("--separate_viewpoints", action="store_true", help="True if LCA should be run independently for left and right. (Legacy - use --separate_by_fields instead)")
    parser.add_argument("--separate_by_fields", nargs="+", help="List of fields to separate runs by, e.g., viewpoint encounter")
    parser.add_argument("--ui_db_path", type=str, default=None, help="Override data.ui_db_path in the LCA input config.")
    parser.add_argument("--max_human_reviews", type=int, default=None, help="Override stability.max_human_reviews in the LCA input config.")
    parser.add_argument("--lca_config", type=str, default=None, help="Path to a custom LCA input config (YAML). Takes precedence over --intra/--inter/--video/--image defaults.")

    args = parser.parse_args()


    if os.path.exists(args.lca_dir) and os.path.isdir(args.lca_dir):
        shutil.rmtree(args.lca_dir) 
    os.makedirs(args.lca_dir)

    # Config for LCA itself -- not input config to LCA
    lca_config = load_config(path_from_file(__file__, "lca_config.yaml"))
    
    lca_alternative_clustering = lca_config["lca_alternative_clustering"]

    # Save to lca dir inside lca
    lca_github_loc = os.path.join(args.lca_dir, "lca_code")
    clone_from_github(lca_github_loc, lca_config["github_lca_url"])

    # OPEN LCA INPUT CONFIG
    if args.lca_config:
        # User-supplied config takes precedence over the mode flags.
        input_config_name = os.path.basename(args.lca_config)
        input_config = load_config(args.lca_config)
        print(f"Using custom LCA config: {args.lca_config}")
    elif args.video:
        input_config_name = "lca_drone.yaml"
        input_config = load_config(path_from_file(__file__, input_config_name))
    elif args.intra:
        input_config_name = "lca_intra.yaml"
        input_config = load_config(path_from_file(__file__, input_config_name))
    elif args.inter:
        input_config_name = "lca_inter.yaml"
        input_config = load_config(path_from_file(__file__, input_config_name))
    else:
        input_config_name = "lca_image.yaml"
        input_config = load_config(path_from_file(__file__, input_config_name))

    # # ADD CONFIG INFO
    input_config["data"]["output_path"] = args.lca_dir
    input_config["data"]["annotation_file"] = args.annots
    input_config["data"]["embedding_file"] = args.embeddings

    # CLI overrides — only applied when explicitly given. Otherwise the
    # value already in the input config (YAML) is authoritative.
    if args.separate_by_fields:
        input_config["data"]["separate_by_fields"] = args.separate_by_fields
        if "separate_viewpoints" in input_config["data"]:
            del input_config["data"]["separate_viewpoints"]
    elif args.separate_viewpoints:
        input_config["data"]["separate_viewpoints"] = args.separate_viewpoints

    if args.ui_db_path is not None:
        input_config["data"]["ui_db_path"] = args.ui_db_path
    if args.max_human_reviews is not None:
        input_config.setdefault("stability", {})["max_human_reviews"] = args.max_human_reviews

    # Resolve effective separation settings from the merged config so all
    # downstream logic uses the same source of truth.
    effective_separate_by_fields = input_config["data"].get("separate_by_fields") or []
    effective_separate_viewpoints = input_config["data"].get("separate_viewpoints", False)

    # input_config["edge_weights"]["verifier_file"] = args.verifiers_probs
    input_config["logging"]["log_file"] = args.log_subunit_file # should append LCA outputs into same log file used by this script


    # WRITE CONFIG FILE INTO LCA
    config_dir = os.path.join(lca_github_loc, lca_config["config_save_path"])
    config_loc = os.path.join(config_dir, input_config_name)

    with open(config_loc, "w") as f:
        yaml.dump(input_config, f)

    # RUN LCA or alternative
    print("Begin LCA Subunit...")
    if lca_alternative_clustering:
        print('run hdbscan')
        subprocess.run(["python3", f"{lca_github_loc}/lca/run_hdbscan.py", "--config", config_loc])
    else:
        print('run lca')
        subprocess.run(["python3", f"{lca_github_loc}/lca/run_clustering_with_save.py", "--config", config_loc])

    output_path = args.lca_dir
    anno_file = args.annots
    uuid_key = input_config["data"]["id_key"]

    # Collect all combinations for unified save
    combinations = []

    if effective_separate_by_fields:
        # New multi-field separation logic
        import glob
        import re

        # Build regex pattern to match and extract field values
        # Pattern: field1-value1_field2-value2_...
        regex_pattern = "_".join([f"{field}-([^_]+)" for field in effective_separate_by_fields])
        regex = re.compile(regex_pattern)

        # Find all directories matching the pattern
        glob_pattern = "_".join([f"{field}-*" for field in effective_separate_by_fields])

        for input_dir in glob.glob(os.path.join(output_path, glob_pattern)):
            if os.path.isdir(input_dir):
                dir_name = os.path.basename(input_dir)
                match = regex.match(dir_name)

                if match:
                    # Extract field values from regex groups
                    field_combo = dict(zip(effective_separate_by_fields, match.groups()))
                    combinations.append((input_dir, field_combo))

    elif effective_separate_viewpoints:
        # Legacy viewpoint separation
        for viewpoint in input_config["data"]["viewpoint_list"]:
            input_dir = os.path.join(output_path, viewpoint)
            combinations.append((input_dir, {"viewpoint": viewpoint}))
    else:
        # No separation
        input_dir = output_path
        combinations.append((input_dir, None))

    # Save unified results
    # if len(combinations) > 0:
    #     save_lca_results_unified(combinations, anno_file, output_path, args.output_prefix, args.output_suffix,
    #                             uuid_key=uuid_key)
    # else:
    #     print("Warning: No clustering results found to save!")

    exit()
