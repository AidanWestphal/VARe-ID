import os

configfile: "config.yaml"

data_is_video = config["data_video"]

exp_name = config["data_dir_out"]

image_dirname = config["image_dirname"]
video_dirname = config["video_dirname"]
annot_dirname = config["annot_dirname"]
model_dirname = config["model_dirname"]

log_dir = os.path.join(exp_name, config["log_dir"])

# for data import
image_out_path = os.path.join(exp_name,config["image_out_file"])
image_dir = os.path.join(exp_name,image_dirname)

video_out_path = os.path.join(exp_name,config["video_out_file"])
video_dir = os.path.join(exp_name,video_dirname)

import_logs = os.path.join(log_dir, config["import_logfile"])

# for detector
dt_dir = os.path.join(exp_name,config["dt_dir"])
annot_dir = os.path.join(exp_name,annot_dirname)
ground_truth_csv = os.path.join(model_dirname,config["ground_truth_csv"])
model_version = config["model_version"]
detector_logs = os.path.join(log_dir, config["dt_logfile"])

vid_annots_filename = "vid_" + config["annots_filename"] + config["model_version"] + ".json"
img_annots_filename = "img_" + config["annots_filename"] + config["model_version"] + ".json"
vid_annots_filtered_filename = "vid_" + config["annots_filtered_filename"] + config["model_version"] + ".json"
img_annots_filtered_filename = "img_" + config["annots_filtered_filename"] + config["model_version"] + ".json"

# SPLIT BASED ON INPUT S.T. DAG HAS TWO PATHS
vid_annots_filtered_path = os.path.join(annot_dir, vid_annots_filtered_filename)
img_annots_filtered_path = os.path.join(annot_dir, img_annots_filtered_filename)

# for species identifier
si_dir = os.path.join(exp_name,config["si_dir"])
predictions_dir = os.path.join(si_dir,config["predictions_dir"])
si_logs = os.path.join(log_dir, config["si_logfile"])

if data_is_video:
    exp_annots_filtered_path = vid_annots_filtered_path
    si_out_path = os.path.join(predictions_dir, vid_annots_filtered_filename + config["si_out_file_end"])
else:
    exp_annots_filtered_path = img_annots_filtered_path
    si_out_path = os.path.join(predictions_dir, img_annots_filtered_filename + config["si_out_file_end"])

# for viewpoint classifier
vc_dir = os.path.join(exp_name,config["vc_dir"])
vc_model_checkpoint = os.path.join(model_dirname,config["vc_model_checkpoint"])
vc_out_path = os.path.join(vc_dir, config["vc_out_file"])
vc_logs = os.path.join(log_dir, config["vc_logfile"])

# for census annotation classifier
cac_dir = os.path.join(exp_name,config["cac_dir"])
cac_model_checkpoint = os.path.join(model_dirname,config["cac_model_checkpoint"])
cac_out_path = os.path.join(cac_dir, config["cac_out_file"])
cac_logs = os.path.join(log_dir, config["cac_logfile"])

# for eda preprocessing
eda_preprocess_path = os.path.join(cac_dir, config["cac_out_filtered"])
eda_logs = os.path.join(log_dir, config["eda_logfile"])

if data_is_video:
    eda_flag = "--video"
else:
    eda_flag = ""

# for frame sampling
fs_dir = os.path.join(exp_name,config["fs_dir"])
fs_out_stage1_path = os.path.join(fs_dir, config["fs_out_stage1_json_file"])
fs_out_final_path = os.path.join(fs_dir, config["fs_out_final_json_file"])
fs_logs = os.path.join(log_dir, config["fs_logfile"])

# DEFINE THE INPUT FOR FRAME SAMPLING BASED ON INPUT TYPE
if not data_is_video:
    fs_out_final_path = eda_preprocess_path

# for miew id embedding generator
mid_dir = os.path.join(exp_name,config["mid_dir"])
mid_model_url = config["mid_model_url"]
mid_out_path = os.path.join(mid_dir, config["mid_out_file"])
mid_logs = os.path.join(log_dir, config["mid_logfile"])

# for lca clustering algorithm
lca_dir = os.path.join(exp_name,config["lca_dir"])
lca_out_dir = os.path.join(exp_name,config["lca_out_dir"])
lca_db_dir = os.path.join(lca_dir, config["lca_db_dir"]) 
lca_logs_path = config["lca_log_file"]
lca_verifiers_path = os.path.join(model_dirname,config["lca_verifiers_probs"])
lca_exp_name = exp_name
lca_logs = os.path.join(log_dir, config["lca_logfile"])

if config["lca_separate_viewpoints"]:
    lca_sep_viewpoint = "--separate_viewpoints"
else:
    lca_sep_viewpoint = ""
if config["use_alternative_clustering"]:
    lca_alg_name = "hdbscan"
else:
    lca_alg_name = "lca"

# for lca post processing
fs_file_name = os.path.basename(fs_out_final_path)
post_dir = os.path.join(exp_name,config["post_dir"])
sep = fs_file_name.rfind(".")
annot_file_no_ext = fs_file_name[:sep].replace(".","")
post_right = os.path.join(lca_out_dir,annot_file_no_ext + config["post_lca_left_end"])
post_left = os.path.join(lca_out_dir,annot_file_no_ext + config["post_lca_right_end"])
post_right_out = os.path.join(post_dir, config["post_lca_right_out"])
post_left_out = os.path.join(post_dir, config["post_lca_left_out"])
post_logs = os.path.join(log_dir, config["post_logfile"])

# TARGET FUNCTION DEFINES WHICH FILES WE WANT TO GENERATE (i.e. DAG follows one path only)
def get_targets():
    targets = list()
    if data_is_video:
        targets.append([video_out_path, vid_annots_filtered_path])
        targets.append([post_right_out,post_left_out])
    else:
        targets.append([image_out_path, img_annots_filtered_path])
        targets.append([post_right,post_left])

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
    log:
        import_logs
    shell:
        "export DYLD_LIBRARY_PATH=/opt/homebrew/opt/zbar/lib && python {input.script} {input.dir} {image_out_path} {output} &> {log}"

rule detector_images:
    input:
        file=image_out_path,
        script="algo/image_detector.py"
    output:
        img_annots_filtered_path
    log:
        detector_logs
    shell:
        "python {input.script} {input.file} {annot_dir} {dt_dir} {ground_truth_csv} {model_version} {img_annots_filename} {img_annots_filtered_filename} &> {log}"

rule import_videos:
    input:
        dir=config["data_dir_in"],
        script="import_videos.py"
    output:
        video_out_path
    log:
        import_logs
    shell:
        "export DYLD_LIBRARY_PATH=/opt/homebrew/opt/zbar/lib && python {input.script} {input.dir} {video_out_path} {output} &> {log}"

rule detector_videos:
    input:
        file=video_out_path,
        script="algo/video_detector.py"
    output:
        vid_annots_filtered_path
    log:
        detector_logs
    shell:
        "python {input.script} {input.file} {annot_dir} {dt_dir} {model_version} {vid_annots_filename} {vid_annots_filtered_filename} &> {log}"

rule species_identifier:
    input:
        file=exp_annots_filtered_path,
        script="algo/species_identifier.py"
    output:
        si_out_path
    log:
        si_logs
    shell:
        "python {input.script} {input.file} {si_dir} {output} &> {log}"
    
rule viewpoint_classifier:
    input:
        file=si_out_path,
        script="algo/viewpoint_classifier.py"
    output:
        vc_out_path
    log:
        vc_logs
    shell:
        "python {input.script} {input.file} {vc_model_checkpoint} {output} &> {log}"

rule ca_classifier:
    input:
        file=vc_out_path,
        script="algo/CA_classifier.py"
    output:
        cac_out_path
    log:
        cac_logs
    shell:
        "python {input.script} {input.file} {cac_model_checkpoint} {output} &> {log}"

rule eda_preprocess:
    input:
        file=cac_out_path,
        script="algo/EDA_preprocess_csv2json.py"
    output:
        eda_preprocess_path
    log:
        eda_logs
    shell:
        "python {input.script} {input.file} {output} {eda_flag} &> {log}"

rule frame_sampling:
    input:
        file=eda_preprocess_path,
        script="algo/frame_sampling.py"
    output:
        json_stage1=fs_out_stage1_path,
        json_final=fs_out_final_path
    log:
        fs_logs
    shell: 
        "python {input.script} {input.file} {output.json_stage1} {output.json_final} &> {log}"

rule miew_id:
    input:
        file=fs_out_final_path,
        script="algo/miew_id.py"
    output:
        mid_out_path
    log:
        mid_logs
    shell: 
        "python {input.script} {input.file} {mid_model_url} {output} &> {log}"

rule lca:
    input:
        annots=fs_out_final_path,
        embeddings=mid_out_path,
        script="algo/lca.py"
    output:
        pr=post_right,
        pl=post_left,
    log:
        lca_logs
    shell: 
        "python {input.script} {lca_dir} {lca_out_dir} {input.annots} {input.embeddings} {lca_verifiers_path} {lca_db_dir} {lca_logs_path} {lca_exp_name} {lca_alg_name} {lca_sep_viewpoint} &> {log}"


rule post:
    input:
        pr=post_right,
        pl=post_left,
        merged=cac_out_path,
        script="algo/LCA_postprocessing_updated.py"
    output:
        right=post_right_out,
        left=post_left_out
    log:
        post_logs
    shell:
        "python {input.script} {input.merged} {image_dir} {input.pr} {input.pl} {post_left_out} {post_right_out} &> {log}"
