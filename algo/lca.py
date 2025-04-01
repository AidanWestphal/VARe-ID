import argparse
import os
import subprocess
import sys
import yaml


def load_config(config_file_path):
    with open(config_file_path, "r") as file:
        config_file = yaml.safe_load(file)
    return config_file


def clone_from_github(directory, repo_url):
    if not os.path.exists(directory) or not os.listdir(directory):
        print(f"Cloning repository {repo_url} into {directory}...")
        subprocess.run(["git", "clone", repo_url, directory], check=True)
        print("Repository cloned successfully...")
    else:
        print("Repository already cloned...")
        

if __name__ == "__main__":
    print("Loading data...")
    parser = argparse.ArgumentParser(
        description="Run LCA clustering"
    )
    parser.add_argument(
        "lca_dir", type=str, help="The directory to save files into."
    )
    parser.add_argument(
        "images", type=str, help="The path to the image directory."
    )
    parser.add_argument(
        "annots", type=str, help="The path to the annotation file."
    )
    parser.add_argument(
        "embeddings", type=str, help="The path to the embeddings file."
    )
    parser.add_argument(
        "verifier_probs", type=str, help="The path to the verifier probabilities."
    )
    parser.add_argument(
        "db_dir", type=str, help="The path to the db directory."
    )
    parser.add_argument(
        "log_file", type=str, help="The path to the log file."
    )

    args = parser.parse_args()

    config = load_config("algo/lca.yaml")
    
    # Save to lca dir inside lca
    lca_github_loc = args.lca_dir + "lca_code"
    clone_from_github(lca_github_loc, config["github_lca_url"])    

    # ADD CONFIG INFO
    config["data"]["images_dir"] = args.images
    config["data"]["annotation_file"] = args.annots 
    config["data"]["embeddings_file"] = args.embeddings
    config["lca"]["db_path"] = args.db_dir
    config["lca"]["verifier_path"] = args.verifier_probs
    # NOTE : Comment this out and set log_file to null in lca.yaml, it will still yield the issue
    config["lca"]["logging"]["log_file"] = args.log_file

    config_loc = os.path.join(lca_github_loc, config["config_save_path"])

    # WRITE CONFIG FILE INTO LCA
    with open(config_loc, "w") as f:
        yaml.dump(config, f)

    # RUN LCA
    # run_loc = os.path.join(lca_github_loc, config["run_path"])
    # subprocess.run([sys.executable, run_loc])

    # TEMPORARY SOLUTION: Directly call script from here
    subprocess.run(["python3", "test_dataset/lca/lca_code/lca/run_drone.py", "--config", config_loc])