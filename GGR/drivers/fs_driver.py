import argparse
import json
import subprocess

def main(args):
    config = json.loads(args.config)

    try:
        subprocess.run(
            f"python -m frame_sampling {config["ia_filtered_out_path"]} {config["fs_stage1_out_path"]} {config["fs_out_path"]} &> {config["fs_logs"]}",
            shell=True, text=True, check=True
        )
    except Exception as e:
        print(e)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Driver script to run the frame sampling component of the pipeline. Performs non-maximum supression to select ideal annotations across a set of tracked annotations."
    )
    parser.add_argument(
        "config",
        type=str,
        help="The built config file.",
    )
    args = parser.parse_args()

    main(args)
