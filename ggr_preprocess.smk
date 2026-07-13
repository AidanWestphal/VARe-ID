import base64
import json

from VAREID.drivers.lca_driver import get_inputs as get_lca_inputs
from VAREID.drivers.lca_driver import get_outputs as get_lca_outputs
from VAREID.drivers.mid_driver import get_inputs as get_mid_inputs
from VAREID.drivers.si_driver import get_inputs as get_si_inputs
from VAREID.libraries.io.workflow_funcs import build_config, generate_ggr_preprocess_targets, encode_config
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
        generate_ggr_preprocess_targets(config)


rule import_images:
    input:
        directory(config["data_dir_in"])
    output:
        config["image_out_path"]
    shell:
        """
        python -m VAREID.drivers.import_image_driver --config {config_str}
        """


rule import_videos:
    input:
        directory(config["data_dir_in"])
    output:
        config["video_out_path"]
    shell:
        """
        python -m VAREID.drivers.import_video_driver --config {config_str}
        """