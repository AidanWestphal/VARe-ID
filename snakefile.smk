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

vid_annots_filename = "vid_" + config["annots_filename"] + config["model_version"]
img_annots_filename = "img_" + config["annots_filename"] + config["model_version"]
vid_annots_filtered_filename = "vid_" + config["annots_filtered_filename"] + config["model_version"]
img_annots_filtered_filename = "img_" + config["annots_filtered_filename"] + config["model_version"]

# SPLIT BASED ON INPUT S.T. DAG HAS TWO PATHS
vid_annots_filtered_path = os.path.join(annot_dir, vid_annots_filtered_filename + ".csv")
img_annots_filtered_path = os.path.join(annot_dir, img_annots_filtered_filename + ".csv")

# for species identifier
si_dir = config["si_dir"]
predictions_dir = config["predictions_dir"]

if data_is_video:
    exp_annots_filtered_path = vid_annots_filtered_path
    si_out_path = os.path.join(predictions_dir, vid_annots_filtered_filename + config["si_out_file_end"])
else:
    exp_annots_filtered_path = img_annots_filtered_path
    si_out_path = os.path.join(predictions_dir, img_annots_filtered_filename + config["si_out_file_end"])

# for viewpoint classifier
vc_dir = config["vc_dir"]
vc_model_checkpoint = config["vc_model_checkpoint"]
vc_out_path = os.path.join(vc_dir, config["vc_out_file"])

# for census annotation classifier
cac_dir = config["cac_dir"]
cac_model_checkpoint = config["cac_model_checkpoint"]
cac_out_path = os.path.join(cac_dir, config["cac_out_filename"]) + ".csv"
eda_preprocess_path = os.path.join(cac_dir, config["cac_out_filename"]) + ".json"

# for frame sampling
fs_dir = config["fs_dir"]
fs_out_stage1_path = os.path.join(fs_dir, config["fs_out_stage1_json_file"])
fs_out_final_path = os.path.join(fs_dir, config["fs_out_final_json_file"])

# for miew id embedding generator
mid_dir = config["mid_dir"]
mid_model_url = config["mid_model_url"]
mid_out_path = os.path.join(mid_dir, config["mid_out_file"])

# for lca clustering algorithm
lca_dir = config["lca_dir"]
lca_db_dir = os.path.join(lca_dir, config["lca_db_dir"]) 
lca_logs_path = os.path.join(lca_dir, config["lca_log_file"])
lca_verifier_path = os.path.join(lca_dir, config["lca_verifier_probs"])
lca_exp_name = config["lca_exp_name"]
if config["lca_separate_viewpoints"]:
    lca_sep_viewpoint = "--separate_viewpoints"
else:
    lca_sep_viewpoint = ""

# TARGET FUNCTION DEFINES WHICH FILES WE WANT TO GENERATE (i.e. DAG follows one path only)
def get_targets():
    targets = list()
    if data_is_video:
        targets.append([video_out_path, vid_annots_filtered_path])
    else:
        targets.append([image_out_path, img_annots_filtered_path])

    targets.append([lca_db_dir])
    return targets

rule all: 
    input:
        get_targets()

rule import_images:
    input:
        dir=config["data_dir_in"],
        script="import_images.py"
    output:
        image_out_path  
    shell:
        # "python {input.script} {input.dir} {img_data_path} {output}"
        "export DYLD_LIBRARY_PATH=/opt/homebrew/opt/zbar/lib && python {input.script} {input.dir} {img_data_path} {output}"

rule detector_images:
    input:
        script="algo/image_detector.py"
    output:
        img_annots_filtered_path
    shell:
        "python {input.script} {image_dir} {annot_dir} {dt_dir} {ground_truth_csv} {model_version} {img_annots_filename} {img_annots_filtered_filename}"

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
        file=video_out_path,
        script="algo/video_detector.py"
    output:
        vid_annots_filtered_path
    shell:
        "python {input.script} {input.file} {annot_dir} {dt_dir} {model_version} {vid_annots_filename} {vid_annots_filtered_filename}"

rule species_identifier:
    input:
        file=exp_annots_filtered_path,
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

rule eda_preprocess:
    input:
        file=cac_out_path,
        script="algo/EDA_preprocess_csv2json.py"
    output:
        eda_preprocess_path
    shell:
        "python {input.script} {input.file} {output}"

rule frame_sampling:
    input:
        file=eda_preprocess_path,
        script="algo/frame_sampling.py"
    output:
        json_stage1=fs_out_stage1_path,
        json_final=fs_out_final_path
    shell: 
        "python {input.script} {image_dir} {input.file} {output.json_stage1} {output.json_final}"

rule miew_id:
    input:
        file=fs_out_final_path,
        script="algo/miew_id.py"
    output:
        mid_out_path
    shell: 
        "python {input.script} {image_dir} {input.file} {mid_model_url} {output}"

rule lca:
    input:
        annots=fs_out_final_path,
        embeddings=mid_out_path,
        script="algo/lca.py"
    output:
        db=lca_db_dir,
        logs=lca_logs_path
    shell: 
        "python {input.script} {lca_dir} {image_dir} {input.annots} {input.embeddings} {lca_verifier_path} {output.db} {output.logs} {lca_exp_name} {lca_sep_viewpoint}"