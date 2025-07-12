import json

from drivers.workflow_funcs import parse_config

# BUILD THE CONFIG FILE
config_dict = parse_config("config.yaml")

# SERIALIZE CONFIG DICT AS STRING
config_str = json.loads(config_dict)

# WORKFLOW IS ORGANIZED BY DRIVERS:

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

rule filteirng:
    input:
        file=cac_out_path,
        script="algo/CA_filtering.py"
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
        script="algo/postprocessing.py"
    output:
        right=post_right_out,
        left=post_left_out
    log:
        post_logs
    shell:
        "python {input.script} {input.merged} {image_dir} {input.pl} {input.pr} {post_left_out} {post_right_out} &> {log}"
