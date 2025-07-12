import argparse
import subprocess

def main(args):
    config = args.config

    input = config["fs_out_file"] if config["data_video"] else config["ia_filtered_out_file"]
    video_flag = "--video" if config["data_video"] else ""
    sv_flag = "--separate_viewpoints" if config["lca_separate_viewpoints"] else ""
    try:
        subprocess.run(
            f"python -m lca {input} {config["mid_out_path"]} {config["lca_verifiers_probs_path"]} {config["lca_dir"]} {config["lca_out_prefix"]} {config["lca_out_suffix"]} {config["lca_logs"]} {video_flag} {sv_flag} &> {config["lca_logs"]}",
            shell=True, text=True, check=True
        )
    except Exception as e:
        print(e)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Driver script to run the LCA component of the pipeline. Clusters annotations."
    )
    parser.add_argument(
        "config",
        type=str,
        help="The built config file.",
    )
    args = parser.parse_args()

    main(args)
