import argparse
import json
import subprocess

def main(args):
    config = json.loads(args.config)

    try:
        subprocess.run(
            f"python -m import_images {config["data_dir_in"]} {config["image_out_path"]} &> {config["import_logs"]}",
            shell=True, text=True, check=True
        )
    except Exception as e:
        print(e)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Driver script to run the import component of the pipeline. Reads in a recursive directory of images or videos and parses them into a json data file. It will also split a video into frames."
    )
    parser.add_argument(
        "config",
        type=str,
        help="The built config file.",
    )
    args = parser.parse_args()

    main(args)
