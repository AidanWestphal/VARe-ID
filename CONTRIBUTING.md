# MODIFYING THE PIPELINE

Being a **modular** pipeline, you have the ability to implement and modify the pipeline's structure. For example, you may choose to completely switch out the species identifier to try a new model or possibly add an entirely new step in the pipeline. This document lays out the process for adding new modular algorithms to the pipeline.

## Algorithm and Driver Main Files

Every component in this pipeline has a folder in `VAREID/algos/` containing all its internal scrips and configuration variables. These scripts are **executable** and should accept parameters via the `argparse` library. Commonly, such input arguments include an input json file, output json file, a path to the model parameters, and checkpoint management variables (frequency to save checkpoints and a path to save checkpoints to). The algorithm should **NOT** depend on the overall pipeline config. Please simplify its arguments to only what's needed.

You will then make a driver function, found in the `VAREID/drivers/` folder. Please copy the format of any other driver function when making yours. This is the script which will actually be called by the snakefile. It will accept the overall config file for the pipeline, perform any small preprocessing of config values that is needed, setup the logger, and execute your component. The general workflow of a driver script will be: decode or build the config file, build the executable command for the component, setup the logger, and execute the logger with the command. In these scripts you will see several API calls from within our pipeline handling config loading and logging. Please copy these formats. 

Note that the driver function will accept either the config file as a path or as an encoded string. This is because the driver function can either be called by the snakefile during a pipeline execution, in which the config has already been built, or it could be ran independently where the config file must still be built. To pass the config over as an argument, the snakefile actually encodes it into a string, which is then decoded by the driver. This logic is all handled by API and can be found in all other driver functions as an example.

## Updating Configs and Library Functions

Now, you will need to update several library calls to handle building the pipeline config. If you added new fields to the annotation json, you will also need to update this. Let us go over every file you may need to update when you add a new component to the pipeline.

### config.json

This is the overall config file for the pipeline. Please copy the format of the other stages of the pipeline when you add new fields to this. Keep it clean. You don't want to put in full file paths or redundant information. You will update a library function later to handle building file paths. Keep this file minimal for cleanliness.

### VAREID/libraries/io/workflow_funcs.py

This file builds `config.json` into file paths. You will need to append to the `build_config` function the path joining for fields you defined in `config.json`. Please follow the convention seen elsewhere in this function and keep it clean. A common operation here is calling `os.path.join` to append the algorithm's output directory to the overall output directory (which is the absolute path to where we are saving all files). For now, you can ignore the target section of this file. The main purpose of target generation is to properly guide the snakefile down a conditional path by identifying a subset of output directories we expect to obtain. This was prominent when we had image and video differences in the pipeline, but should mostly be obsolete for now.

### VAREID/libraries/io/format_funcs.py

This file handles loading and saving of annotations to our COCO json format. As such, if you need to add or modify the schema of our json file (such as adding a new field), you should add the string to the lists at the top of this file. Do not worry about the rename mappings. These are there to allow us to rename important ID fields to match other tables (like `image_uuid` to `uuid`) for joining. You should not need to change this, but it may come up.

### main_algorithm.smk

To add your new rule, insert a rule into the snakefile. You must specify an input and output file so it knows where to go in the pipeline. You must then update the rule which follows this component to expect your new rule's output as its input. Snakemake automatically resolves rules into a DAG (Directed Acyclic Graph) to reach the final output, which will trace through your new component. Your rule will call your new driver function and pass in the config string. Please follow the formatting of all other rules in this file. You'll also need to import your new driver function.

## API Consistency

Most of the algorithm components and driver scripts follow a specific set of API standards within this repository. For example, we always use the functions in `format_funcs.py` to build and save json files. There is also a common formatting for utilizing checkpoint management, which will save and resume long inference tasks. My best advice to you is to study the other component scripts before writing your own and mimic their practices. Try your best to use the API calls within this repository for consistency.
