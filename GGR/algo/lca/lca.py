import argparse
import os
import subprocess
import sys
import shutil
import yaml

import pandas as pd

from GGR.util.io.format_funcs import clone_from_github, load_config, load_json, save_json, split_dataframe, join_dataframe_dict
from GGR.util.utils import path_from_file


def save_lca_results(input_dir, anno_file, output_path, prefix, suffix, viewpoint=None, uuid_key="annot_uuid"):
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

    # Filter annotations based on viewpoint if provided
    if viewpoint is not None:
        filtered_annotations = [
            ann for ann in data['annotations']
            if ann.get('viewpoint', '').strip().lower() == viewpoint.strip().lower()
        ]
        print(f"Filtered {len(filtered_annotations)} annotations with viewpoint='{viewpoint}' "
            f"out of {len(data['annotations'])}")
    else:
        filtered_annotations = data['annotations']

    # Add LCA_clustering_id to each annotation
    for ann in filtered_annotations:
        ann['LCA_clustering_id'] = uuid_to_cluster.get(ann[uuid_key], None)
    
    # Build output path
    if viewpoint is not None:
        output_filename = f"{prefix}_{viewpoint}_{suffix}.json"
    else:
        output_filename = f"{prefix}_{suffix}.json"
    output_path = os.path.join(output_path, output_filename)

    # Save final result with same categories/images, modified annotations
    result_dict = split_dataframe(pd.DataFrame(filtered_annotations))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_json(result_dict, output_path)
    

if __name__ == "__main__":
    print("Loading data...")
    parser = argparse.ArgumentParser(description="Run LCA clustering")
    parser.add_argument("annots", type=str, help="The path to the annotation file.")
    parser.add_argument("embeddings", type=str, help="The path to the embeddings file.")
    parser.add_argument("verifiers_probs", type=str, help="The path to the verifier probabilities.")
    parser.add_argument("lca_dir", type=str, help="The directory to save files into.")
    parser.add_argument("output_prefix", type=str, help="The prefix for the output annotation file format.")
    parser.add_argument("output_suffix", type=str, help="The suffix for the output annotation file format.")
    parser.add_argument("log_subunit_file", type=str, help="The path to the log file for the LCA algorithm itself.")
    parser.add_argument("log_file", type=str, help="The path to the log file.")
    parser.add_argument("--video", action="store_true", help="True if LCA should run on the video config file.")
    parser.add_argument("--separate_viewpoints", action="store_true", help="True if LCA should be run independently for left and right.")
    parser.add_argument("--image_dir", type=str, default=None, help="The path to the image directory. Optional, only required if input data doesn't have an image_path field.")

    args = parser.parse_args()

    # Get abs path to current dir
    # TODO: FIX
    current_dir = "get_abs_path()"

    # Config for LCA itself -- not input config to LCA
    lca_config = load_config(path_from_file(__file__, "lca_config.yaml"))
    
    lca_alternative_clustering = lca_config["lca_alternative_clustering"]

    # Save to lca dir inside lca
    lca_github_loc = os.path.join(args.lca_dir, "lca_code")
    clone_from_github(lca_github_loc, lca_config["github_lca_url"])

    # OPEN LCA INPUT CONFIG
    if args.video:
        input_config_name = "lca_drone.yaml"
        input_config = load_config(path_from_file(__file__, input_config_name))
    else:
        input_config_name = "lca_image.yaml"
        input_config = load_config(path_from_file(__file__, input_config_name))

    # ADD CONFIG INFO
    input_config["data"]["images_dir"] = args.image_dir # or None
    input_config["data"]["output_path"] = args.lca_dir
    input_config["data"]["annotation_file"] = args.annots
    input_config["data"]["embedding_file"] = args.embeddings
    input_config["data"]["separate_viewpoints"] = args.separate_viewpoints
    input_config["edge_weights"]["verifier_file"] = args.verifiers_probs
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
        subprocess.run(["python3", f"{lca_github_loc}/lca/run.py", "--config", config_loc])

    output_path = args.lca_dir
    anno_file = args.annots

    if args.separate_viewpoints:
        for viewpoint in input_config["data"]["viewpoint_list"]:
            input_dir = os.path.join(output_path, viewpoint)
            save_lca_results(input_dir, anno_file, output_path, args.output_prefix, args.output_suffix, viewpoint=viewpoint, uuid_key=input_config["data"]["id_key"])
    else:
        input_dir = output_path
        save_lca_results(input_dir, anno_file, output_path, args.output_prefix, args.output_suffix, uuid_key=input_config["data"]["id_key"])

    exit()
