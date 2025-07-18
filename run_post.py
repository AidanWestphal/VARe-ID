import argparse
from multiprocessing import Process
import os
from pathlib import Path

from GGR.util.io.format_funcs import load_config
from GGR.util.io.logging import log_subprocess, setup_logging
from GGR.util.io.workflow_funcs import build_config

def main(args):
    config = build_config(load_config(args.config_path))

    Path(os.path.dirname(config["post_db_path"])).mkdir(parents=True, exist_ok=True)
    
    # READ INTERACTION TYPE
    interaction_mode = config["interaction_mode"]
    if interaction_mode not in ["database", "ipywidgets", "console"]:
        interaction_mode = "database"

    post_command = f'python -u -m GGR.algo.postprocessing.postprocessing {config["image_dir"]} {config["post_left_in_path"]} {config["post_right_in_path"]} {config["post_left_out_path"]} {config["post_right_out_path"]} --db {config["post_db_path"]} --interaction_mode {interaction_mode}'
    gui_command = f'python -u -m GGR.util.ui.gui --db {config["post_db_path"]} --allowed_dir {config["data_dir_out"]}'

    # MODE (UI)
    if interaction_mode == "database":
        post_logger = setup_logging(config["post_logs"])
        gui_logger = setup_logging(config["gui_logs"])

        # THREADDING TO RUN SIMULTANEOUSLY
        post_process = Process(target=log_subprocess, args=(post_command, post_logger))
        gui_process = Process(target=log_subprocess, args=(gui_command, gui_logger))

        # STAT POST AND WAIT FOR DB PATH TO APPEAR
        post_process.start()
        
        while not os.path.exists(config["post_db_path"]):
            # IF THE PROCESS STOPPED (we never ran GUI...)
            if not post_process.is_alive():
                post_logger.critical(f"POST PROCESSING EXITED BEFORE CREATING A DB FILE. SKIPPING GUI CREATION.")
                gui_logger.critical(f"POST PROCESSING EXITED BEFORE CREATING A DB FILE. SKIPPING GUI CREATION.")
                exit(1)

        # DB created, open it
        gui_process.start()
        
        # Join at end
        post_process.join()
        # If GUI is alive, kill it. We are DONE!
        if gui_process.is_alive():
            gui_logger.info(f"POST PROCESSING TERMINATING. EXITING GUI.")
            gui_process.terminate()
            gui_process.join()

    # IPYWIDGETS AND CONSOLE INTERACTION
    else:
        post_logger = setup_logging(config["post_logs"])
        log_subprocess(post_command, post_logger)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Driver script to run the postprocessing component of the pipeline. Only use for video data. Resolves ambiguities via human decision."
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="A path to the config file to load. Config file MUST be structured like config.yaml!",
    )
    args = parser.parse_args()

    main(args)
