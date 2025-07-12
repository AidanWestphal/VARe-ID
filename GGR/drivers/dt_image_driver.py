import argparse
import subprocess

def main(args):
    config = args.config

    try:
        subprocess.run(
            f"python -m image_detector {config["image_out_path"]} {config["annot_dir"]} {config["dt_dir"]} {config["dt_gt_path"]} {config["dt_model"]} {config["dt_pre_filtering_path"]} {config["dt_out_path"]} &> {config["dt_logs"]}",
            shell=True, text=True, check=True
        )
    except Exception as e:
        print(e)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Driver script to run the detection component of the pipeline. Generates base annotations from the set of images (or video frames)."
    )
    parser.add_argument(
        "config",
        type=str,
        help="The built config file.",
    )
    args = parser.parse_args()

    main(args)
