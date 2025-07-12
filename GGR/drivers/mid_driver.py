import argparse
import subprocess

def main(args):
    config = args.config
    
    input = config["fs_out_path"] if config["data_video"] else config["ia_filtered_out_path"]
    try:
        subprocess.run(
            f"python ../algo/miew_id/miew_id.py {input} {config["mid_model"]} {config["mid_out_path"]} &> {config["mid_logs"]}",
            shell=True, text=True, check=True
        )
    except Exception as e:
        print(e)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Driver script to run the miew-id embedding component of the pipeline. Generates embeddings for annotations."
    )
    parser.add_argument(
        "config",
        type=str,
        help="The built config file.",
    )
    args = parser.parse_args()

    main(args)
