import argparse
import subprocess

def main(args):
    config = args.config

    try:
        subprocess.run(
            f"python -m species_identifier {config["dt_out_path"]} {config["si_dir"]} {config["si_out_path"]} &> {config["si_logs"]}",
            shell=True, text=True, check=True
        )
    except Exception as e:
        print(e)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Driver script to run the species identification component of the pipeline. Classifies annotations by species (grevys, plains, or neither)."
    )
    parser.add_argument(
        "config",
        type=str,
        help="The built config file.",
    )
    args = parser.parse_args()

    main(args)
