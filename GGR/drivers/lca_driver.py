import argparse

from GGR.util.io.logging import log_subprocess, setup_logging
from GGR.util.io.workflow_funcs import decode_config

def get_inputs(config):
    inputs = [config["mid_out_path"]]

    if config["data_video"]:
        inputs.append(config["fs_out_path"])
    else:
        inputs.append(config["ia_filtered_out_path"])

    return inputs


def get_outputs(config):
    if config["lca_separate_viewpoints"]:
        outputs = [config["post_left_in_path"], config["post_right_in_path"]]
    else:
        outputs = [config["lca_out_path"]]

    return outputs


def main(args):
    config = decode_config(args.config)

    input = config["fs_out_path"] if config["data_video"] else config["ia_filtered_out_path"]
    video_flag = "--video" if config["data_video"] else ""
    sv_flag = "--separate_viewpoints" if config["lca_separate_viewpoints"] else ""

    command = f'python -u -m GGR.algo.lca.lca {input} {config["mid_out_path"]} {config["lca_verifiers_probs_path"]} {config["lca_dir"]} {config["lca_out_prefix"]} {config["lca_out_suffix"]} {config["lca_subunit_logs"]} {config["lca_logs"]} {video_flag} {sv_flag}'

    logger = setup_logging(config["lca_logs"])
    log_subprocess(command, logger)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Driver script to run the LCA component of the pipeline. Clusters annotations."
    )
    parser.add_argument(
        "config",
        type=str,
        help="The built config file.",
    )
    args = parser.parse_args()

    main(args)
