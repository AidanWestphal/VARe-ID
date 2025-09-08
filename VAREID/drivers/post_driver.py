import argparse
from multiprocessing import Process, Pipe
import os
from pathlib import Path
import signal

from VAREID.libraries.io.format_funcs import load_config
from VAREID.libraries.io.logging import log_subprocess, setup_logging
from VAREID.libraries.io.workflow_funcs import build_config

def main(args):
    config = build_config(load_config(args.config_path))

    Path(os.path.dirname(config["post_db_path"])).mkdir(parents=True, exist_ok=True)
    
    # READ INTERACTION TYPE
    interaction_mode = config["interaction_mode"]
    if interaction_mode not in ["database", "ipywidgets", "console"]:
        interaction_mode = "database"

    post_command = f'python -u -m VAREID.algo.postprocessing.postprocessing {config["image_dir"]} {config["post_left_in_path"]} {config["post_right_in_path"]} {config["post_left_out_path"]} {config["post_right_out_path"]} --db {config["post_db_path"]} --interaction_mode {interaction_mode}'
    gui_command = f'python -u -m VAREID.algo.postprocessing.gui --db {config["post_db_path"]} --allowed_dir {config["data_dir_out"]}'

    # MODE (UI)
    if interaction_mode == "database":
        post_logger = setup_logging(config["post_logs"])
        gui_logger = setup_logging(config["gui_logs"])

        # PID PIPE FOR GUI
        p_conn, c_conn = Pipe()

        # THREADDING TO RUN SIMULTANEOUSLY
        post_process = Process(target=log_subprocess, args=(post_command, post_logger))
        gui_process = Process(target=log_subprocess, args=(gui_command, gui_logger), kwargs={"conn": c_conn})

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
            pid = p_conn.recv()
            gui_logger.info(f"TERMINATING GUI PROCESS (PID {pid}) & FREEING PORT.")
            # Kill the process directly
            os.kill(pid, signal.SIGTERM)
            # Join the killed process
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
