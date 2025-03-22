import os

configfile: "config.yaml"

data_is_video = config["data_video"]

db_dir = config["db_dir"]
image_dirname = config["image_dirname"]
video_dirname = config["video_dirname"]
annot_dirname = config["annot_dirname"]

# for data import
img_data_path = db_dir + config["image_out_file"]
image_dir = db_dir + image_dirname + "/"
image_out_path = os.path.join(db_dir, config["image_out_file"])

video_data_path = db_dir + config["video_out_file"]
video_dir = db_dir + video_dirname + "/"
video_out_path = os.path.join(db_dir, config["video_out_file"])

# for detector
dt_dir = config["dt_dir"]
annot_dir = db_dir + annot_dirname + "/"
ground_truth_csv = config["ground_truth_csv"]
model_version = config["model_version"]
annots_filename = config["annots_filename"] + config["model_version"]
annots_filtered_filename = config["annots_filtered_filename"] + config["model_version"]

# SPLIT BASED ON INPUT S.T. DAG HAS TWO PATHS
vid_annots_filtered_path = os.path.join(annot_dir, "vid_" + annots_filtered_filename + ".csv")
img_annots_filtered_path = os.path.join(annot_dir, "img_" + annots_filtered_filename + ".csv")


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
mid_out_json_path = os.path.join(mid_dir, config["mid_out_json_file"])
mid_out_pkl_path = os.path.join(mid_dir, config["mid_out_pkl_file"])

# TARGET FUNCTION DEFINES WHICH FILES WE WANT TO GENERATE (i.e. DAG follows one path only)
def get_targets():
    targets = list()
    if data_is_video:
        targets.append([video_out_path, vid_annots_filtered_path])
    else:
        targets.append([image_out_path, img_annots_filtered_path])

    targets.append([mid_out_json_path,mid_out_pkl_path])
    return targets

rule all: 
    input:
        get_targets()

# rule import_data:
#     input:
#         dir=config["data_dir_in"],
#         script="import_data.py"
#     output:
#         import_out_path
#     shell:
#         "python {input.script} {input.dir} {img_data_path} {output}"

rule import_images:
    input:
        dir=config["data_dir_in"],
        script="import_images.py"
    output:
        image_out_path  
    shell:
        "export DYLD_LIBRARY_PATH=/opt/homebrew/opt/zbar/lib && python {input.script} {input.dir} {img_data_path} {output}"

rule detector_images:
    input:
        script="algo/image_detector.py"
    output:
        img_annots_filtered_path
    shell:
        "python {input.script} {image_dir} {annot_dir} {dt_dir} {ground_truth_csv} {model_version} {annots_filename} {annots_filtered_filename}"

rule import_videos:
    input:
        dir=config["data_dir_in"],
        script="import_videos.py"
    output:
        video_out_path
    shell:
        "export DYLD_LIBRARY_PATH=/opt/homebrew/opt/zbar/lib && python {input.script} {input.dir} {video_data_path} {output}"

rule detector_videos:
    input:
        dir=config["data_dir_in"],
        script="algo/video_detector.py"
    output:
        vid_annots_filtered_path
    shell:
        "python {input.script} {input.dir} {image_dir} {annot_dir} {dt_dir} {ground_truth_csv} {model_version} {annots_filename} {annots_filtered_filename}"

# FUNCTION TO JOIN INPUT AND DETECTION PATHS
def determine_path(wildards):
        if data_is_video:
            return vid_annots_filtered_path
        else:
            return img_annots_filtered_path

rule species_identifier:
    input:
        file=determine_path,
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
        json=mid_out_json_path, 
        pkl=mid_out_pkl_path
    shell: 
        "python {input.script} {image_dir} {input.file} {mid_model_url} {output.json} {output.pkl}"
