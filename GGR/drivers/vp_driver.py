import argparse
import subprocess

def main(args):
    config = args.config

    try:
        subprocess.run(
            f"python -m viewpoint_classifier {config["si_out_path"]} {config["vp_model_path"]} {config["vp_out_path"]} &> {config["vp_logs"]}",
            shell=True, text=True, check=True
        )
    except Exception as e:
        print(e)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Driver script to run the viewpoint classification component of the pipeline. Classifies an annotation by its viewpoint (up, left, right, front, back, or any combination of those)."
    )
    parser.add_argument(
        "config",
        type=str,
        help="The built config file.",
    )
    args = parser.parse_args()

    main(args)
