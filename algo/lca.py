import argparse
import os
import subprocess
import sys
import yaml

from util.format_funcs import load_config, load_json, save_json, split_dataframe, join_dataframe


def clone_from_github(directory, repo_url):
    if not os.path.exists(directory) or not os.listdir(directory):
        print(f"Cloning repository {repo_url} into {directory}...")
        subprocess.run(["git", "clone", repo_url, directory], check=True)
        print("Repository cloned successfully...")
    else:
        print("Repository already cloned...")


def save_lca_results(input_dir, anno_file, output_dir, viewpoint=None):
    clustering_file = os.path.join(input_dir, "clustering.json")
    node2uuid_file = os.path.join(input_dir, "node2uuid_file.json")

    data = load_json(anno_file)
    df = join_dataframe(data)

    # Optionally filter by viewpoint
    if viewpoint is not None:
        df = df[df['viewpoint'] == viewpoint]

    # Load clustering results
    clusters = load_json(clustering_file)
    node2uuid = load_json(node2uuid_file)

    # Map UUIDs to cluster IDs
    uuid_to_cluster = {}
    for cluster_id, nodes in clusters.items():
        for node in nodes:
            uuid = node2uuid.get(str(node))
            if uuid:
                uuid_to_cluster[uuid] = cluster_id

    # Ensure UUID column exists
    if 'uuid' not in df.columns:
        df['uuid'] = df['uuid_x']

    # Assign cluster IDs
    df['LCA_clustering_id'] = df['uuid'].map(uuid_to_cluster).where(df['uuid'].isin(uuid_to_cluster), None)

    # Modify output file name
    suffix = f"LCA_{viewpoint}" if viewpoint else "LCA"
    name, ext = os.path.splitext(os.path.basename(anno_file))
    output_filename = f"{name}_{suffix}{ext}"
    output_path = os.path.join(output_dir, output_filename)

    result_dict = split_dataframe(df)

    # Save result
    os.makedirs(output_dir, exist_ok=True)
    save_json(result_dict, output_path)


if __name__ == "__main__":
    print("Loading data...")
    parser = argparse.ArgumentParser(description="Run LCA clustering")
    parser.add_argument("lca_dir", type=str, help="The directory to save files into.")
    parser.add_argument("--image_dir", type=str, default=None, help="The path to the image directory.")
    parser.add_argument("output_dir", type=str, help="The directory to save files into.")
    parser.add_argument("annots", type=str, help="The path to the annotation file.")
    parser.add_argument("embeddings", type=str, help="The path to the embeddings file.")
    parser.add_argument(
        "verifier_probs", type=str, help="The path to the verifier probabilities."
    )
    parser.add_argument("db_path", type=str, help="The path to the db .")
    parser.add_argument("log_file", type=str, help="The path to the log file.")
    parser.add_argument("exp_name", type=str, help="The name of the experiment")
    parser.add_argument("alg_name", type=str, help="The name of the clustering algorithm: lca or hdbscan")
    parser.add_argument(
        "--separate_viewpoints",
        action="store_true",
        help="True if LCA should be run independently for left and right.",
    )

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    config = load_config(os.path.join(script_dir, "lca.yaml"))

    # Save to lca dir inside lca
    lca_github_loc = os.path.join(args.lca_dir, "lca_code")
    clone_from_github(lca_github_loc, config["github_lca_url"])

    # ADD CONFIG INFO
    config["data"]["images_dir"] = args.image_dir
    config["data"]["output_dir"] = args.output_dir
    config["data"]["annotation_file"] = args.annots
    config["data"]["embedding_file"] = args.embeddings
    config["data"]["separate_viewpoints"] = args.separate_viewpoints
    config["data"]["ui_db_path"] = args.db_path
    config["lca"]["db_path"] = args.db_path
    config["lca"]["verifier_path"] = args.verifier_probs
    config["exp_name"] = args.exp_name
    # NOTE : Comment this out and set log_file to null in lca.yaml, it will still yield the issue
    config["lca"]["logging"]["log_file"] = os.path.join(args.output_dir, args.log_file)

    os.makedirs(args.output_dir, exist_ok=True)
    open(config["lca"]["logging"]["log_file"], 'a').close()

    config_loc = os.path.join(lca_github_loc, config["config_save_path"])

    # WRITE CONFIG FILE INTO LCA
    with open(config_loc, "w") as f:
        yaml.dump(config, f)

    # RUN LCA or alternative
    if args.alg_name == "hdbscan":
        print('run hdbscan')
        subprocess.run(["python3", f"{lca_github_loc}/lca/run_hdbscan.py", "--config", config_loc])
    else:
        subprocess.run(["python3", f"{lca_github_loc}/lca/run_drone.py", "--config", config_loc])


    output_dir = args.output_dir
    anno_file = args.annots

    if args.separate_viewpoints:
        for viewpoint in config["data"]["viewpoint_list"]:
            input_dir = os.path.join(output_dir, args.exp_name, viewpoint)
            save_lca_results(input_dir, anno_file, output_dir, viewpoint=viewpoint)
    else:
        input_dir = os.path.join(output_dir, args.exp_name)
        save_lca_results(input_dir, anno_file, output_dir)

    exit()
