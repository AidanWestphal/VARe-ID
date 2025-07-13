import argparse
import subprocess

from GGR.drivers.workflow_funcs import decode_config

def main(args):
    config = decode_config(args.config)

    try:
        subprocess.run(
            f'python -m IA_classifier {config["vc_out_path"]} {config["ia_model_path"]} {config["ia_out_path"]} &> {config["ia_logs"]}',
            shell=True, text=True, check=True
        )
    except Exception as e:
        print(e)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Driver script to run the identifiable annotation classification component of the pipeline. Classifies an annotation by its quality/ability to be identified."
    )
    parser.add_argument(
        "config",
        type=str,
        help="The built config file.",
    )
    args = parser.parse_args()

    main(args)
