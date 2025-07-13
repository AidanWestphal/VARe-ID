import json

from GGR.drivers.mid_driver import get_inputs as get_mid_inputs
from GGR.drivers.workflow_funcs import parse_config, generate_targets

# BUILD THE CONFIG FILE
config = parse_config("config.yaml")

# SERIALIZE CONFIG DICT AS STRING
config_str = json.loads(config)

# WORKFLOW IS ORGANIZED BY DRIVERS

rule all: 
    input:
        generate_targets()


rule import_images:
    input:
        directory(config["data_dir_in"])
    output:
        config["image_out_path"]
    shell:
        f"python -m import_image_driver {config_str}"


rule import_videos:
    input:
        directory(config["data_dir_in"])
    output:
        config["video_out_path"]
    shell:
        f"python -m video_image_driver {config_str}"


rule detect_images:
    input:
        config["image_out_path"]
    output:
        config["dt_out_path"]
    shell:
        f"python -m dt_image_driver {config_str}"
    

rule detect_videos:
    input:
        config["video_out_path"]
    output:
        config["dt_out_path"]
    shell:
        f"python -m dt_video_driver {config_str}"


rule species_identification:
    input:
        config["dt_out_path"]
    output:
        config["si_out_path"]
    shell:
        f"python -m si_driver {config_str}"


rule viewpoint_classification:
    input:
        config["si_out_path"]
    output:
        config["vp_out_path"]
    shell:
        f"python -m vp_driver {config_str}"


rule ia_classification:
    input:
        config["vp_out_path"]
    output:
        config["ia_out_path"]
    shell:
        f"python -m iac_driver {config_str}"

rule ia_filtering:
    input:
        config["ia_out_path"]
    output:
        config["ia_filtered_out_path"]
    shell:
        f"python -m iaf_driver {config_str}"


rule frame_sampling:
    input:
        config["ia_filtered_out_path"]
    output:
        config["fs_out_path"]
    shell:
        f"python -m fs_driver {config_str}"


rule miew_id:
    input:
        *get_mid_inputs(config)
    output:
        config["mid_out_path"]
    shell:
        f"python -m fs_driver {config_str}"


rule lca:
    input:
        *get_lca_inputs(config)
    output:
        *get_lca_outputs(config)
    shell:
        f"python -m lca {config_str}"