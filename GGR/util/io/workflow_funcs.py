import base64
import json
import os

from GGR.util.io.format_funcs import load_config


def decode_config(code):
    '''
    Decodes a config dictionary from str.

    Parameters:
        code (str): The encoded config dictionary as a str.
    
    Returns:
        config (dict): The decoded config dictionary.
    '''

    encoded_bytes = code.encode('utf-8')
    decoded_bytes = base64.b64decode(encoded_bytes)
    decoded_str = decoded_bytes.decode('utf-8')

    return json.loads(decoded_str)


def encode_config(config):
    '''
    Encodes a config dictionary into a str thats passable in shell.

    Parameters:
        config (dict): The config dictionary.
    
    Returns:
        code (str): The encoded config dictionary as a str.
    '''

    config_str = json.dumps(config)
    original_bytes = config_str.encode('utf-8')
    encoded_bytes = base64.b64encode(original_bytes)

    return encoded_bytes.decode('utf-8')


def generate_targets(config):
    '''
    Generates the snakemake target files for the all rule s.t. the pipeline follows the proper 
    workflow depending on its mode (video or image mode).

    Parameters:
        config (dict): The dictionary format of config.yaml with all important paths built.

    Returns:
        targets (list): The list of targets (file paths) that snakemake must generate.
    '''

    targets = [config["si_out_path"], config["vc_out_path"], config["ia_out_path"]]

    if config["data_video"]:
        targets.append([config["video_out_path"], config["dt_video_out_path"], config["fs_out_path"]])
    else:
        targets.append([config["image_out_path"], config["dt_image_out_path"]])

    if config["lca_separate_viewpoints"]:
        targets.append([config["post_left_in_path"], config["post_right_in_path"]])
    else:
        targets.append([config["lca_out_path"]])

    return targets


def parse_config(filepath):
    '''
    Parses the general config.yaml file and builds all important fields used in the pipeline.

    This function reads the config file (as described in config.yaml) as a dictionary, builds 
    all necessary paths, and returns the dictionary with these new paths saved into it.

    Parameters:
        filepath (str): The path to config.yaml in the root directory.

    Returns:
        config (dict): The dictionary format of config.yaml with all important paths built.
    '''

    config = load_config(filepath)

    # MAJOR BUILDER FIELDS FOR ALL STEPS IN PIPELINE
    model_dir = config["model_dirname"]
    out_dir = config["data_dir_out"]

    # MAJOR OUTPUT DIRECTORIES
    image_dir = os.path.join(out_dir, config["image_dirname"])
    log_dir = os.path.join(out_dir, config["log_dirname"])

    # IMPORT STEP
    image_out_path = os.path.join(out_dir, config["image_out_file"])
    video_out_path = os.path.join(out_dir, config["video_out_file"])
    import_logs = os.path.join(log_dir, config["import_logfile"])

    # DETECTION STEP
    dt_dir = os.path.join(out_dir, config["dt_dirname"])
    dt_video_out_path = os.path.join(dt_dir, config["dt_video_out_file"])
    dt_image_out_path = os.path.join(dt_dir, config["dt_image_out_file"])
    dt_logs = os.path.join(log_dir, config["dt_logfile"])

    # If a GT file was provided, build it to a path, otherwise set path to none
    dt_gt_path = os.path.join(dt_dir, config["dt_gt_file"]) if config["dt_gt_file"] is not None else None
    dt_filtered_out_path =  os.path.join(dt_dir, config["dt_filtered_out_file"]) if config["dt_filtered_out_file"] is not None else None

    # SPECIES IDENTIFICATION STEP
    si_dir = os.path.join(out_dir, config["si_dirname"])
    si_out_path = os.path.join(si_dir, config["si_out_file"])
    si_logs = os.path.join(log_dir, config["si_logfile"])

    # VIEWPOINT CLASSIFICATION STEP
    vc_dir = os.path.join(out_dir, config["vc_dirname"])
    vc_model_path = os.path.join(model_dir, config["vc_model"])
    vc_out_path = os.path.join(vc_dir, config["vc_out_file"])
    vc_logs = os.path.join(log_dir, config["vc_logfile"])

    # IDENTIFIABLE ANNOTATION CLASSIFICATION STEP
    ia_dir = os.path.join(out_dir, config["ia_dirname"])
    ia_model_path = os.path.join(model_dir, config["ia_model"])
    ia_out_path = os.path.join(ia_dir, config["ia_out_file"])
    ia_filtered_out_path = os.path.join(ia_dir, config["ia_filtered_out_file"])
    ia_logs = os.path.join(log_dir, config["ia_logfile"])

    # FRAME SAMPLING STEP
    fs_dir = os.path.join(out_dir, config["fs_dirname"])
    fs_out_path = os.path.join(fs_dir, config["fs_out_file"])
    fs_logs = os.path.join(log_dir, config["fs_logfile"])

    # Optional stage1 frame sampling output, generate path iff not none
    fs_stage1_out_path = os.path.join(fs_dir, config["fs_stage1_out_file"]) if config["fs_stage1_out_file"] is not None else None

    # MIEWID EMBEDDING STEP
    mid_dir = os.path.join(out_dir, config["mid_dirname"])
    mid_out_path = os.path.join(mid_dir, config["mid_out_file"])
    mid_logs = os.path.join(log_dir, config["mid_logfile"])

    # LCA STEP
    lca_dir = os.path.join(out_dir, config["lca_dirname"])
    lca_verifiers_probs_path = os.path.join(model_dir, config["lca_verifiers_probs"])
    lca_subunit_logs = os.path.join(log_dir, config["lca_subunit_logfile"])
    lca_logs = os.path.join(log_dir, config["lca_logfile"])

    # In video mode, post expects specifically left and right viewpoints. Build the paths to these files here:
    # Format is lca_dir/{prefix}_{left/right}_{suffix}.json
    post_left_in_path = os.path.join(lca_dir, f'{config["lca_out_prefix"]}_left_{config["lca_out_suffix"]}.json')
    post_right_in_path = os.path.join(lca_dir, f'{config["lca_out_prefix"]}_right_{config["lca_out_suffix"]}.json')
    # In image mode, we expect the following output to the entire pipeline:
    lca_out_path = os.path.join(lca_dir, f'{config["lca_out_prefix"]}_{config["lca_out_suffix"]}.json')

    # POST PROCESSING STEP
    post_dir = os.path.join(out_dir, config["post_dirname"])
    post_left_out_path = os.path.join(post_dir, config["post_left_out_file"])
    post_right_out_path = os.path.join(post_dir, config["post_right_out_file"])
    post_logs = os.path.join(log_dir, config["post_logfile"])

    # REINSERT ALL NEWLY-BUILT PATHS INTO CONFIG DICTIONARY
    config["image_dir"] = image_dir
    config["log_dir"] = log_dir
    config["image_out_path"] = image_out_path
    config["video_out_path"] = video_out_path
    config["import_logs"] = import_logs
    config["dt_dir"] = dt_dir
    config["dt_video_out_path"] = dt_video_out_path
    config["dt_image_out_path"] = dt_image_out_path
    config["dt_logs"] = dt_logs
    config["dt_gt_path"] = dt_gt_path
    config["dt_filtered_out_path"] = dt_filtered_out_path
    config["si_dir"] = si_dir
    config["si_out_path"] = si_out_path
    config["si_logs"] = si_logs
    config["vc_dir"] = vc_dir
    config["vc_model_path"] = vc_model_path
    config["vc_out_path"] = vc_out_path
    config["vc_logs"] = vc_logs
    config["ia_dir"] = ia_dir
    config["ia_model_path"] = ia_model_path
    config["ia_out_path"] = ia_out_path
    config["ia_filtered_out_path"] = ia_filtered_out_path
    config["ia_logs"] = ia_logs
    config["fs_dir"] = fs_dir
    config["fs_out_path"] = fs_out_path
    config["fs_logs"] = fs_logs
    config["fs_stage1_out_path"] = fs_stage1_out_path
    config["mid_dir"] = mid_dir
    config["mid_out_path"] = mid_out_path
    config["mid_logs"] = mid_logs
    config["lca_dir"] = lca_dir
    config["lca_verifiers_probs_path"] = lca_verifiers_probs_path
    config["post_left_in_path"] = post_left_in_path
    config["post_right_in_path"] = post_right_in_path
    config["lca_subunit_logs"] = lca_subunit_logs
    config["lca_logs"] = lca_logs
    config["lca_out_path"] = lca_out_path
    config["post_dir"] = post_dir
    config["post_left_out_path"] = post_left_out_path
    config["post_right_out_path"] = post_right_out_path
    config["post_logs"] = post_logs

    return config
