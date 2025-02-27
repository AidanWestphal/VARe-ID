import os

configfile: "config.yaml"

db_dir = config["db_dir"]
image_dirname = config["image_dirname"]
annot_dirname = config["annot_dirname"]

# for data import
img_data_path = db_dir + config["import_out_file"]
image_dir = db_dir + image_dirname + "/"
import_out_path = os.path.join(db_dir, config["import_out_file"])

# for detector
dt_dir = config["dt_dir"]
annot_dir = db_dir + annot_dirname + "/"
ground_truth_csv = config["ground_truth_csv"]
model_version = config["model_version"]
annots_filename = config["annots_filename"] + config["model_version"]
annots_filtered_filename = config["annots_filtered_filename"] + config["model_version"]
annots_filtered_path = os.path.join(annot_dir, annots_filtered_filename + ".csv")

# for species identifier
si_dir = config["si_dir"]
predictions_dir = config["predictions_dir"]
si_out_path = os.path.join(predictions_dir, annots_filtered_filename + config["si_out_file_end"])

# for viewpoint classifier
vc_dir = config["vc_dir"]
vc_model_checkpoint = config["vc_model_checkpoint"]
vc_out_path = os.path.join(vc_dir, config["vc_out_file"])

# for census annotation classifier
cac_dir = config["cac_dir"]
cac_model_checkpoint = config["cac_model_checkpoint"]
cac_out_path = os.path.join(cac_dir, config["cac_out_file"])

# for miew id embedding generator
mid_dir = config["mid_dir"]
mid_model_url = config["mid_model_url"]
mid_out_path = os.path.join(mid_dir, config["mid_out_file"])

rule all: 
    input:
        si_out_path,
        mid_out_path

# rule import_data:
#     input:
#         dir=config["data_dir_in"],
#         script="import_data.py"
#     output:
#         import_out_path
#     shell:
#         "python {input.script} {input.dir} {img_data_path} {output}"

rule import_data:
    input:
        dir=config["data_dir_in"],
        script="import_data.py"
    output:
        import_out_path  # Define this variable appropriately
    shell:
        "export DYLD_LIBRARY_PATH=/opt/homebrew/opt/zbar/lib && python {input.script} {input.dir} {img_data_path} {output}"


rule detector:
    input:
        file=import_out_path,
        script="algo/detector.py"
    output:
        annots_filtered_path
    shell:
        "python {input.script} {image_dir} {annot_dir} {dt_dir} {ground_truth_csv} {model_version} {annots_filename} {annots_filtered_filename}"

rule species_identifier:
    input:
        file=annots_filtered_path,
        script="algo/species_identifier.py"
    output:
        si_out_path
    shell:
        "python {input.script} {image_dir} {input.file} {si_dir} {output}"
    
rule viewpoint_classifier:
    input:
        file=si_out_path,
        script="algo/viewpoint_classifier.py"
    output:
        vc_out_path
    shell:
        "python {input.script} {image_dir} {input.file} {vc_model_checkpoint} {output}"

rule ca_classifier:
    input:
        file=vc_out_path,
        script="algo/CA_classifier.py"
    output:
        cac_out_path
    shell:
        "python {input.script} {image_dir} {input.file} {cac_model_checkpoint} {output}"

rule miew_id:
    input:
        file=cac_out_path,
        script="algo/miew_id.py"
    output:
        mid_out_path
    shell: 
        "python {input.script} {image_dir} {input.file} {mid_model_url} {output}"
