import argparse
import subprocess

def main(args):
    config = args.config

    flag = "--video" if config["data_video"] else ""
    try:
        subprocess.run(
            f"python -m IA_filtering {config["ia_out_path"]} {config["ia_filtered_out_path"]} {flag} &> {config["ia_logs"]}",
            shell=True, text=True, check=True
        )
    except Exception as e:
        print(e)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Driver script to run the identifiable annotation filtering component of the pipeline. Filters out all annotations not suitable for identification and simplifies viewpoint."
    )
    parser.add_argument(
        "config",
        type=str,
        help="The built config file.",
    )
    args = parser.parse_args()

    main(args)
