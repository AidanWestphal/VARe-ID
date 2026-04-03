import base64
import json

from VAREID.drivers.eg_driver import get_inputs as get_eg_inputs
from VAREID.drivers.eg_driver import get_outputs as get_eg_outputs
from VAREID.drivers.lca_driver import get_inputs as get_lca_inputs
from VAREID.drivers.lca_driver import get_outputs as get_lca_outputs
from VAREID.drivers.mid_driver import get_inputs as get_mid_inputs
from VAREID.drivers.representative_driver import get_inputs as get_rep_inputs
from VAREID.drivers.representative_driver import get_outputs as get_rep_outputs
from VAREID.drivers.forward_clustering_driver import get_inputs as get_fc_inputs
from VAREID.drivers.forward_clustering_driver import get_outputs as get_fc_outputs
from VAREID.drivers.si_driver import get_inputs as get_si_inputs
from VAREID.libraries.io.workflow_funcs import build_config, generate_targets, encode_config
from VAREID.libraries.utils import path_from_file

# Default configfile, can supply alternative with --configfile
configfile: "config.yaml"

# BUILD THE CONFIG FILE
config = build_config(config)

# SERIALIZE CONFIG DICT AS STRING (and add quotes to either side s.t. its passed as a param)
config_str = encode_config(config)

# WORKFLOW IS ORGANIZED BY DRIVERS

rule all: 
    input:
        generate_targets(config)


rule detect_images:
    input:
        config["image_out_path"]
    output:
        config["dt_image_out_path"]
    shell:
        "python -m VAREID.drivers.dt_image_driver --config {config_str}"
    

rule detect_videos:
    input:
        config["video_out_path"]
    output:
        config["dt_video_out_path"]
    shell:
        "python -m VAREID.drivers.dt_video_driver --config {config_str}"


rule species_identification:
    input:
        *get_si_inputs(config)
    output:
        config["si_out_path"]
    shell:
        "python -m VAREID.drivers.si_driver --config {config_str}"


rule viewpoint_classification:
    input:
        config["si_out_path"]
    output:
        config["vc_out_path"]
    shell:
        "python -m VAREID.drivers.vc_driver --config {config_str}"


rule ia_classification:
    input:
        config["vc_out_path"]
    output:
        config["ia_out_path"]
    shell:
        "python -m VAREID.drivers.iac_driver --config {config_str}"


rule ia_filtering:
    input:
        config["ia_out_path"]
    output:
        config["ia_filtered_out_path"]
    shell:
        "python -m VAREID.drivers.iaf_driver --config {config_str}"


rule id_region:
    input:
        config["ia_filtered_out_path"]
    output:
        config["idr_out_path"]
    shell:
        "python -m VAREID.drivers.idr_driver --config {config_str}"


rule id_region_filtering:
    input:
        config["idr_out_path"]
    output:
        config["idr_filtered_out_path"]
    shell:
        "python -m VAREID.drivers.idrf_driver --config {config_str}"


rule frame_sampling:
    input:
        config["idr_filtered_out_path"]
    output:
        config["fs_out_path"]
    shell:
        "python -m VAREID.drivers.fs_driver --config {config_str}"


rule miew_id:
    input:
        *get_mid_inputs(config)
    output:
        config["mid_out_path"]
    shell:
        "python -m VAREID.drivers.mid_driver --config {config_str}"


rule encounter_grouping:
    input:
        *get_eg_inputs(config)
    output:
        *get_eg_outputs(config)
    shell:
        "python -m VAREID.drivers.eg_driver --config {config_str}"


rule intra_lca:
    input:
        *get_lca_inputs(config, intra=True)
    output:
        *get_lca_outputs(config, intra=True)
    shell:
        "python -m VAREID.drivers.lca_driver --config {config_str} --intra"


rule representative_selection:
    input:
        *get_rep_inputs(config)
    output:
        *get_rep_outputs(config)
    shell:
        "python -m VAREID.drivers.representative_driver --config {config_str}"


rule inter_lca:
    input:
        *get_lca_inputs(config, inter=True)
    output:
        *get_lca_outputs(config, inter=True)
    shell:
        "python -m VAREID.drivers.lca_driver --config {config_str} --inter"


rule forward_clustering:
    input:
        *get_fc_inputs(config)
    output:
        *get_fc_outputs(config)
    shell:
        "python -m VAREID.drivers.forward_clustering_driver --config {config_str}"

