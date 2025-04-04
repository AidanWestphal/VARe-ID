#!/usr/bin/env python3

import argparse
import os
import sys
import time
import yaml
import pandas as pd
import csv
import re
import ast
import json
import pickle
import math
from collections import defaultdict
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from PIL import Image

try:
    get_ipython().run_line_magic("matplotlib", "inline")
except:
    pass

###############################################################################
#                         GLOBAL DECISION LOG + TXT                           #
###############################################################################
GLOBAL_DECISION_LOG = {
    "split_auto_merge_count": 0,
    "split_auto_no_merge_count": 0,
    "split_ambiguous_count": 0,
    "time_auto_merge_count": 0,
    "time_auto_no_merge_count": 0,
    "time_ambiguous_count": 0,
}

# >>> ADDED OR MODIFIED <<<
# Maintain a global list of per-threshold results
GLOBAL_RUN_RESULTS = []

def write_results_to_txt(filename, metrics, decision_log, config, threshold_lower, threshold_upper):
    """
    Simplified final summary with clean ID assignment display
    """
    with open(filename, "a") as f:
        # Threshold information header
        f.write("\n" + "="*50 + "\n")
        if threshold_upper is None:  # Single mode
            f.write(f"=== RUN WITH SINGLE THRESHOLD: {threshold_lower:.2f} ===\n")
        else:  # Dual mode
            f.write(f"=== RUN WITH THRESHOLDS: lower={threshold_lower:.2f}, upper={threshold_upper:.2f} ===\n")
        f.write("="*50 + "\n\n")
        
        # Configuration summary (unchanged)
        f.write("=== Configuration Summary ===\n\n")
        f.write(f"Left Viewpoint Input: {config['json']['left']['input']}\n")
        f.write(f"Right Viewpoint Input: {config['json']['right']['input']}\n")
        f.write(f"Metadata CSV: {config['csv']['metadata']}\n")
        
        emb_cfg = config.get("embedding", {})
        f.write("\nEmbedding Settings:\n")
        f.write(f"  Embedding file: {emb_cfg.get('file', 'None')}\n")
        f.write(f"  Distance metric: {emb_cfg.get('distance_metric', 'euclidean')}\n")
        f.write(f"  Mode: {emb_cfg.get('mode', 'dual')}\n")
        if threshold_upper is None:
            f.write(f"  Threshold: {threshold_lower}\n")
        else:
            f.write(f"  Lower threshold: {threshold_lower}\n")
            f.write(f"  Upper threshold: {threshold_upper}\n")
        f.write(f"  Threshold step: {emb_cfg.get('threshold_step', 0.1)}\n")

        # Final metrics
        f.write("\n=== Final Results Summary ===\n\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1 Score:  {metrics['f1']:.4f}\n")
        f.write(f"\nTrue Positives: {metrics['true_positives']}\n")
        f.write(f"False Positives: {metrics['false_positives']}\n")
        f.write(f"False Negatives: {metrics['false_negatives']}\n")
        f.write(f"\nTotal ground truth individuals: {metrics['total_gt_individuals']}\n")
        f.write(f"Total predicted clusters: {metrics['total_pred_clusters']}\n")

        # Decision counts
        f.write("\n=== Decision Counts ===\n")
        f.write("\nSplit Stage:\n")
        f.write(f"  Auto Merge:    {decision_log['split_auto_merge_count']}\n")
        f.write(f"  Auto No Merge: {decision_log['split_auto_no_merge_count']}\n")
        f.write(f"  Ambiguous:     {decision_log['split_ambiguous_count']}\n")

        f.write("\nTime Stage:\n")
        f.write(f"  Auto Merge:    {decision_log['time_auto_merge_count']}\n")
        f.write(f"  Auto No Merge: {decision_log['time_auto_no_merge_count']}\n")
        f.write(f"  Ambiguous:     {decision_log['time_ambiguous_count']}\n")

        # Final ID assignments
        f.write("\n=== Final ID Assignment Summary ===\n")
        
        # Get final data
        final_left = metrics.get('final_left', metrics['time_stage_left'])
        final_right = metrics.get('final_right', metrics['time_stage_right'])
        
        # Left viewpoint
        f.write("\nLeft Viewpoint:\n")
        left_grouped = group_annotations_by_LCA_with_viewpoint(final_left, 'left')
        for cl in sorted(left_grouped.keys()):
            anns = left_grouped[cl]
            gt_ids = {str(ann.get('individual_id', 'None')) for ann in anns}
            assigned_ids = {str(ann.get('final_id', 'None')) for ann in anns}
            f.write(f"Cluster {cl}: GT IDs: {gt_ids} | Assigned IDs: {assigned_ids}\n")
        
        # Right viewpoint
        f.write("\nRight Viewpoint:\n")
        right_grouped = group_annotations_by_LCA_with_viewpoint(final_right, 'right')
        for cl in sorted(right_grouped.keys()):
            anns = right_grouped[cl]
            gt_ids = {str(ann.get('individual_id', 'None')) for ann in anns}
            assigned_ids = {str(ann.get('final_id', 'None')) for ann in anns}
            f.write(f"Cluster {cl}: GT IDs: {gt_ids} | Assigned IDs: {assigned_ids}\n")
        
        # ID distribution
        f.write("\n=== ID Distribution ===\n")
        id_counts = defaultdict(int)
        for ann in final_left['annotations'] + final_right['annotations']:
            if 'final_id' in ann:
                id_counts[ann['final_id']] += 1
        
        for id_, count in sorted(id_counts.items()):
            f.write(f"ID {id_}: {count} annotations\n")

###############################################################################
#                          STAGE 1: MERGE DETECTION CSVs                      #
###############################################################################

def merge_csv_files(file1_path, file2_path, output_path):
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    merged_df = pd.concat([df1, df2], ignore_index=True)
    merged_df = merged_df.sort_values('timestamp')
    merged_df.to_csv(output_path, index=False)
    print(f"[Stage 1] Merged detection CSV files into: {output_path}")
    print(f"[Stage 1] Total rows: {len(merged_df)}")

###############################################################################
#                  STAGE 2: MERGE CSVs WITH BOUNDING BOXES                    #
###############################################################################

def parse_bbox_string_xyxy(bbox_str):
    s = bbox_str.strip()
    s = re.sub(r'[\[\]\(\)]', '', s)
    parts = s.split(',')
    try:
        return tuple(int(p.strip()) for p in parts)
    except:
        return None

def parse_bbox_string_xywh(bbox_str):
    s = bbox_str.strip()
    s = re.sub(r'[\[\]\(\)]', '', s)
    parts = s.split(',')
    try:
        coords = [int(p.strip()) for p in parts]
        if len(coords) != 4:
            return None
        x1, y1, w, h = coords
        return (x1, y1, x1 + w, y1 + h)
    except:
        return None

def merge_csvs_with_bbox(timestamps_csv, metadata_csv, output_csv):
    dict_ts = {}
    with open(timestamps_csv, 'r', newline='', encoding='utf-8') as f1:
        reader1 = csv.DictReader(f1)
        ts_fieldnames = reader1.fieldnames or []
        for row in reader1:
            ts_filename = row.get('frame_name')
            if not ts_filename:
                continue
            ts_bbox_str = row.get('bbox')
            if not ts_bbox_str:
                continue
            ts_bbox = parse_bbox_string_xyxy(ts_bbox_str)
            if ts_bbox is None:
                continue
            key = (ts_filename, ts_bbox)
            dict_ts[key] = row

    merged_rows = []
    combined_fieldnames = set(ts_fieldnames)

    with open(metadata_csv, 'r', newline='', encoding='utf-8') as f2:
        reader2 = csv.DictReader(f2)
        metadata_fieldnames = reader2.fieldnames or []
        for col in metadata_fieldnames:
            combined_fieldnames.add(col)
        for row in reader2:
            md_filename = row.get('file_name')
            if not md_filename:
                continue
            md_bbox_str = row.get('bbox')
            if not md_bbox_str:
                continue
            md_bbox = parse_bbox_string_xywh(md_bbox_str)
            if md_bbox is None:
                continue
            key = (md_filename, md_bbox)
            if key in dict_ts:
                ts_row = dict_ts[key]
                merged_dict = {}
                merged_dict.update(ts_row)
                merged_dict.update(row)
                merged_rows.append(merged_dict)

    combined_fieldnames = list(combined_fieldnames)
    with open(output_csv, 'w', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, fieldnames=combined_fieldnames)
        writer.writeheader()
        for row in merged_rows:
            writer.writerow(row)

    print(f"[Stage 2] Merged CSV with bounding boxes written to: {output_csv}")
    print(f"[Stage 2] Total merged rows: {len(merged_rows)}")

###############################################################################
#             STAGE 3: UPDATE JSON ANNOTATIONS WITH TIMESTAMPS                #
###############################################################################

def parse_bbox(bbox_str):
    try:
        return list(ast.literal_eval(bbox_str))
    except:
        return None

def make_comparable_dict_from_csv(row, file_to_uuid):
    """
    Create a dictionary for matching JSON annotations, including individual_id if present.
    """
    def safe_int(x):
        try:
            return int(x.strip())
        except:
            return None

    def safe_float(x):
        try:
            return float(x.strip())
        except:
            return None

    tracking_id = safe_int(row.get("tracking_id", ""))
    confidence = safe_float(row.get("confidence", ""))
    detection_class = safe_int(row.get("detection_class", ""))
    CA_score = safe_float(row.get("CA_score", ""))
    individual_id = safe_int(row.get("individual_id", ""))

    species = row.get("species", "").strip()
    bbox = parse_bbox(row.get("bbox", ""))
    image_uuid = file_to_uuid.get(row.get("file_name", "").strip(), None)
    timestamp = row.get("timestamp", "").strip()

    comparable_fields = {
        "image_uuid": image_uuid,
        "tracking_id": tracking_id,
        "confidence": confidence,
        "detection_class": detection_class,
        "species": species,
        "bbox": bbox,
        "CA_score": CA_score,
        "individual_id": individual_id,
    }
    return comparable_fields, timestamp

def make_comparable_dict_from_json(ann):
    bbox_list = list(ann["bbox"]) if ann.get("bbox") else None
    return {
        "image_uuid": ann["image_uuid"],
        "tracking_id": ann["tracking_id"],
        "confidence": ann["confidence"],
        "detection_class": ann["detection_class"],
        "species": ann["species"],
        "bbox": bbox_list,
        "CA_score": ann["CA_score"],
        "individual_id": ann.get("individual_id", None),
    }

def update_json_with_timestamp(json_input, csv_input, json_output):
    with open(json_input, "r") as f:
        data = json.load(f)

    file_to_uuid = {img["file_name"]: img["uuid"] for img in data["images"]}

    csv_common_list = []
    with open(csv_input, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            comp_fields, ts = make_comparable_dict_from_csv(row, file_to_uuid)
            if comp_fields["image_uuid"] is not None:
                csv_common_list.append((comp_fields, ts))

    updated_annotations = 0
    for (csv_fields, csv_timestamp) in csv_common_list:
        for ann in data["annotations"]:
            ann_fields = make_comparable_dict_from_json(ann)
            if ann_fields == csv_fields:
                ann["timestamp"] = csv_timestamp
                updated_annotations += 1
                break

    with open(json_output, "w") as f:
        json.dump(data, f, indent=4)

    print(f"[Stage 3] Updated JSON written to: {json_output}")
    print(f"[Stage 3] Number of annotations updated: {updated_annotations}")

###############################################################################
#                      HELPERS: PRINT CLUSTER SUMMARIES                       #
###############################################################################

def print_cluster_summary(data):
    annotations = data.get("annotations", [])
    total = len(annotations)
    grouped = defaultdict(list)
    for ann in annotations:
        key = ann.get("LCA_clustering_id")
        if key is not None:
            grouped[key].append(ann)
    unique = len(grouped)
    print(f"\nTotal count of annotations: {total}")
    print(f"Count of unique clustering IDs: {unique}\n")
    for cluster in sorted(grouped.keys()):
        print(f"LCA_clustering_id: {cluster}")
        for ann in grouped[cluster]:
            ca = ann.get("CA_score", 0)
            ind_id = ann.get("individual_id", None)
            print(f"  Tracking ID: {ann.get('tracking_id')}, Individual ID: {ind_id}, CA_score: {ca:.8f}")
        print()

def print_viewpoint_cluster_mapping(data, viewpoint):
    annotations = data.get("annotations", [])
    grouped = defaultdict(set)
    for ann in annotations:
        key = ann.get("LCA_clustering_id")
        if key is not None:
            grouped[key].add(ann.get("tracking_id"))
    print(f"\n{viewpoint.capitalize()} Viewpoint Cluster to Tracking ID Mapping:")
    for cluster in sorted(grouped.keys()):
        print(f"  Cluster {cluster}_{viewpoint}: Tracking IDs: {grouped[cluster]}")
    print()

###############################################################################
#                        DISTANCE & MERGING LOGIC                             #
###############################################################################

def euclidean_distance(v1, v2):
    return math.dist(v1, v2)

def cosine_distance(v1, v2):
    """
    Returns 1 - cos_sim, which can be in [0,2].
    """
    dot = sum(a*b for (a,b) in zip(v1, v2))
    mag1 = math.sqrt(sum(a*a for a in v1))
    mag2 = math.sqrt(sum(a*a for a in v2))
    if mag1 == 0 or mag2 == 0:
        return 1.0
    return 1.0 - (dot/(mag1*mag2))

def bounded_cosine_distance(v1, v2):
    """
    Bounded distance in [0,1].
    Identical => 0.0, Opposite => 1.0
    """
    dot = sum(a*b for (a,b) in zip(v1, v2))
    norm1 = math.sqrt(sum(a*a for a in v1))
    norm2 = math.sqrt(sum(a*a for a in v2))
    if norm1 == 0 or norm2 == 0:
        return 1.0
    cos_sim = dot / (norm1 * norm2)  # in [-1,1]
    return 0.5 * (1.0 - cos_sim)

def load_embeddings(pickle_path):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    embedding_matrix = data[0]
    annotation_uuids = data[1]
    uuid_to_index = {}
    for i, uid in enumerate(annotation_uuids):
        uuid_to_index[uid] = i
    return embedding_matrix, uuid_to_index

def get_cluster_best_annotation_embedding(annotations, embedding_matrix, uuid_to_index):
    best_ann = max(annotations, key=lambda x: x.get("CA_score", 0.0))
    best_uuid = best_ann["uuid"]
    idx = uuid_to_index.get(best_uuid, None)
    if idx is not None:
        return embedding_matrix[idx], best_ann
    return None, best_ann

def update_cluster_merge(grouped_annotations, source_cluster, target_cluster):
    grouped_annotations[target_cluster].extend(grouped_annotations[source_cluster])
    for ann in grouped_annotations[source_cluster]:
        ann['LCA_clustering_id'] = target_cluster
    del grouped_annotations[source_cluster]

def pairwise_verification_with_embedding(
    grouped_annotations, cluster1, cluster2, data_dict,
    embedding_matrix, uuid_to_index,
    threshold_lower, threshold_upper,
    stage
):
    """
    Return True if a cluster merge happened, else False.
    """
    if cluster1 not in grouped_annotations or cluster2 not in grouped_annotations:
        return False
    if not grouped_annotations[cluster1] or not grouped_annotations[cluster2]:
        return False

    emb1, best_ann1 = get_cluster_best_annotation_embedding(grouped_annotations[cluster1], embedding_matrix, uuid_to_index)
    emb2, best_ann2 = get_cluster_best_annotation_embedding(grouped_annotations[cluster2], embedding_matrix, uuid_to_index)

    if emb1 is None or emb2 is None:
        print(f"[{stage.capitalize()}] WARNING: Missing embedding for {cluster1} or {cluster2}. Skipping merge.")
        return False

    metric = data_dict.get("distance_metric", "euclidean")
    if metric == "cosine":
        dist = bounded_cosine_distance(emb1, emb2)  # strictly in [0,1]
    else:
        dist = euclidean_distance(emb1, emb2)       # unbounded if not normalized

    print(f"[{stage.capitalize()}] Checking {cluster1} vs {cluster2}, distance={dist:.4f} (metric={metric})")

    if dist < threshold_lower:
        print(f"[{stage.capitalize()}] => distance < {threshold_lower}, auto merge.")
        update_cluster_merge(grouped_annotations, cluster2, cluster1)
        if stage == "split":
            GLOBAL_DECISION_LOG["split_auto_merge_count"] += 1
        else:
            GLOBAL_DECISION_LOG["time_auto_merge_count"] += 1
        return True
    elif dist > threshold_upper:
        print(f"[{stage.capitalize()}] => distance > {threshold_upper}, auto no-merge.")
        if stage == "split":
            GLOBAL_DECISION_LOG["split_auto_no_merge_count"] += 1
        else:
            GLOBAL_DECISION_LOG["time_auto_no_merge_count"] += 1
        return False
    else:
        # ambiguous => check ground truth
        same_gt = (best_ann1.get("individual_id") == best_ann2.get("individual_id"))
        if same_gt:
            print(f"[{stage.capitalize()}] => ambiguous but same GT => merge.")
            update_cluster_merge(grouped_annotations, cluster2, cluster1)
            if stage == "split":
                GLOBAL_DECISION_LOG["split_ambiguous_count"] += 1
            else:
                GLOBAL_DECISION_LOG["time_ambiguous_count"] += 1
            return True
        else:
            print(f"[{stage.capitalize()}] => ambiguous but different GT => no merge.")
            if stage == "split":
                GLOBAL_DECISION_LOG["split_ambiguous_count"] += 1
            else:
                GLOBAL_DECISION_LOG["time_ambiguous_count"] += 1
            return False

###############################################################################
#                      STAGE 4: SPLIT CLUSTER VERIFICATION                    #
###############################################################################

def group_annotations_by_LCA_all(data):
    annotations = data['annotations']
    grouped = defaultdict(list)
    for ann in annotations:
        lca = ann.get('LCA_clustering_id')
        if lca is not None:
            grouped[lca].append(ann)
    return grouped

def consistency_check_with_embeddings(grouped_left, grouped_right, data_left, data_right,
                                      embedding_matrix, uuid_to_index,
                                      threshold_lower, threshold_upper, distance_metric,
                                      stage_name="split"):
    changed = False
    # 1) Left viewpoint
    while True:
        left_tracking = defaultdict(set)
        for lca, anns in grouped_left.items():
            for ann in anns:
                left_tracking[ann['tracking_id']].add(lca)

        any_merge = False
        for tid, clusters in left_tracking.items():
            if len(clusters) > 1:
                cl_list = list(clusters)
                for i in range(len(cl_list) - 1):
                    c1 = cl_list[i]
                    c2 = cl_list[i+1]
                    merged = pairwise_verification_with_embedding(
                        grouped_left, c1, c2,
                        {"distance_metric": distance_metric},
                        embedding_matrix, uuid_to_index,
                        threshold_lower, threshold_upper,
                        stage_name
                    )
                    if merged:
                        changed = True
                        any_merge = True
                        break
                if any_merge:
                    break
        if not any_merge:
            break

    # 2) Right viewpoint
    while True:
        right_tracking = defaultdict(set)
        for lca, anns in grouped_right.items():
            for ann in anns:
                right_tracking[ann['tracking_id']].add(lca)

        any_merge = False
        for tid, clusters in right_tracking.items():
            if len(clusters) > 1:
                cl_list = list(clusters)
                for i in range(len(cl_list) - 1):
                    c1 = cl_list[i]
                    c2 = cl_list[i+1]
                    merged = pairwise_verification_with_embedding(
                        grouped_right, c1, c2,
                        {"distance_metric": distance_metric},
                        embedding_matrix, uuid_to_index,
                        threshold_lower, threshold_upper,
                        stage_name
                    )
                    if merged:
                        changed = True
                        any_merge = True
                        break
                if any_merge:
                    break
        if not any_merge:
            break

    return changed

###############################################################################
#                      STAGE 6: TIME-OVERLAP VERIFICATION                     #
###############################################################################

def parse_timestamp(ts_str):
    """Exact replica of original timestamp parser"""
    date_part, time_part = ts_str.split(" ")
    first_comma = time_part.find(",")
    if first_comma != -1:
        time_part = time_part[:first_comma] + "." + time_part[first_comma+1:]
        time_part = time_part.replace(",", "")
    fixed = f"{date_part} {time_part}"
    return datetime.strptime(fixed, "%Y-%m-%d %H:%M:%S.%f")

def intervals_overlap(start1, end1, start2, end2):
    """Original overlap check without threshold"""
    return start1 <= end2 and start2 <= end1

def find_clusters_with_no_time_overlap(grouped_annotations):
    """Modified to exactly match original behavior"""
    cluster_intervals = defaultdict(dict)
    
    # Build time intervals per cluster
    for cl, anns in grouped_annotations.items():
        tid_to_times = defaultdict(list)
        for ann in anns:
            if 'timestamp' not in ann:
                continue
            try:
                dt = parse_timestamp(ann['timestamp'])
                tid_to_times[ann['tracking_id']].append(dt)
            except:
                continue
        
        for tid, times in tid_to_times.items():
            if not times:
                continue
            times.sort()
            cluster_intervals[cl][tid] = (times[0], times[-1])

    # Find non-overlapping cluster pairs
    no_overlap = []
    clusters = sorted(cluster_intervals.keys())
    
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            c1 = clusters[i]
            c2 = clusters[j]
            found_overlap = False
            
            # Check all tracking ID pairs between clusters
            for (start1, end1) in cluster_intervals[c1].values():
                for (start2, end2) in cluster_intervals[c2].values():
                    if intervals_overlap(start1, end1, start2, end2):
                        found_overlap = True
                        break
                if found_overlap:
                    break
            
            if not found_overlap:
                no_overlap.append((c1, c2))
    
    return no_overlap

def time_overlap_verification_with_embeddings(
    grouped_annotations, data_dict,
    embedding_matrix, uuid_to_index,
    threshold_lower, threshold_upper,
    distance_metric
):
    """Updated verification function"""
    no_overlap_pairs = find_clusters_with_no_time_overlap(grouped_annotations)
    
    if not no_overlap_pairs:
        print("[Time Stage] All clusters have some time overlap; no merges to check.")
        return

    print(f"[Time Stage] Found {len(no_overlap_pairs)} cluster pair(s) with NO time overlap.")
    for (c1, c2) in no_overlap_pairs:
        pairwise_verification_with_embedding(
            grouped_annotations, c1, c2,
            data_dict,
            embedding_matrix, uuid_to_index,
            threshold_lower, threshold_upper,
            "time"
        )

###############################################################################
#                STAGE 5: CLUSTER EQUIVALENCE & STAGE 7: ID ASSIGN            #
###############################################################################

def group_annotations_by_LCA_with_viewpoint(data, viewpoint):
    grouped = defaultdict(list)
    for ann in data['annotations']:
        lca = ann.get('LCA_clustering_id')
        if lca is not None:
            grouped[f"{lca}_{viewpoint}"].append(ann)
    return grouped

def check_cluster_equivalence(grouped_left, grouped_right):
    print("\nStage 5: Checking Cluster Equivalence (No 'individual_id' changes)")
    print("================================================================\n")

    tracking_left = {}
    tracking_right = {}

    for cl, anns in grouped_left.items():
        for ann in anns:
            tracking_left[ann['tracking_id']] = cl

    for cl, anns in grouped_right.items():
        for ann in anns:
            tracking_right[ann['tracking_id']] = cl

    for left_cl, anns in grouped_left.items():
        left_tids = {ann['tracking_id'] for ann in anns}
        rcs = {tracking_right[tid] for tid in left_tids if tid in tracking_right}
        names = {rc.rsplit('_',1)[0] for rc in rcs if rc}
        if len(names) > 1:
            print(f"Cluster {left_cl.rsplit('_',1)[0]} in Left VP spans multiple clusters in Right VP: {names}")
        else:
            print(f"Cluster {left_cl.rsplit('_',1)[0]} in Left VP ~ Right VP cluster(s): {names}")

    for right_cl, anns in grouped_right.items():
        right_tids = {ann['tracking_id'] for ann in anns}
        lcs = {tracking_left[tid] for tid in right_tids if tid in tracking_left}
        names = {lc.rsplit('_',1)[0] for lc in lcs if lc}
        if len(names) > 1:
            print(f"Cluster {right_cl.rsplit('_',1)[0]} in Right VP spans multiple clusters in Left VP: {names}")
        else:
            print(f"Cluster {right_cl.rsplit('_',1)[0]} in Right VP ~ Left VP cluster(s): {names}")

    print("\nStage 5 Completed: Cluster Equivalence Check Done.\n")

def assign_ids_after_equivalence_check(data_left, data_right, output_file="results_summary.txt"):
    print("\n--- Stage 7: Assigning IDs after equivalence check ---")
    
    # Open the output file in append mode
    with open(output_file, "a") as f:
        f.write("\n--- Stage 7: Assigning IDs after equivalence check ---\n")
        
        # Group annotations by LCA_clustering_id with viewpoint suffix
        grouped_left = group_annotations_by_LCA_with_viewpoint(data_left, 'left')
        grouped_right = group_annotations_by_LCA_with_viewpoint(data_right, 'right')
        
        # Create tracking_id to cluster mappings
        left_tracking = {}
        right_tracking = {}
        
        for cl, anns in grouped_left.items():
            for ann in anns:
                left_tracking[ann['tracking_id']] = cl
                
        for cl, anns in grouped_right.items():
            for ann in anns:
                right_tracking[ann['tracking_id']] = cl
        
        # Build equivalence groups
        equivalence_groups = []
        processed_clusters = set()
        
        # First pass: find matching clusters between viewpoints
        for left_cl, left_anns in grouped_left.items():
            if left_cl in processed_clusters:
                continue
                
            # Find all tracking IDs in this left cluster
            left_tids = {ann['tracking_id'] for ann in left_anns}
            
            # Find corresponding right clusters
            right_clusters = set()
            for tid in left_tids:
                if tid in right_tracking:
                    right_clusters.add(right_tracking[tid])
            
            if right_clusters:
                # Create an equivalence group
                group = {left_cl} | right_clusters
                equivalence_groups.append(group)
                processed_clusters.update(group)
        
        # Second pass: handle remaining unprocessed clusters
        all_left_clusters = set(grouped_left.keys())
        all_right_clusters = set(grouped_right.keys())
        unprocessed = (all_left_clusters | all_right_clusters) - processed_clusters
        
        # Assign IDs
        individual_id = 1
        
        # Process equivalence groups first
        for group in equivalence_groups:
            # Get all annotations in this equivalence group
            all_anns = []
            for cl in group:
                if cl.endswith('_left'):
                    all_anns.extend(grouped_left[cl])
                else:
                    all_anns.extend(grouped_right[cl])
            
            # Assign the same individual ID to all
            for ann in all_anns:
                if 'assigned_id' not in ann:  # Only assign if not already assigned
                    ann['assigned_id'] = individual_id
                ann['final_id'] = individual_id  # Always set final_id
            
            msg = f"Assigned ID {individual_id} to equivalent clusters: {[cl.split('_')[0] for cl in group]}"
            print(msg)
            f.write(msg + "\n")
            individual_id += 1
        
        # Process remaining unprocessed clusters
        for cl in unprocessed:
            if cl.endswith('_left'):
                anns = grouped_left[cl]
                prefix = "left"
            else:
                anns = grouped_right[cl]
                prefix = "right"
            
            # Assign temporary ID
            cluster_num = cl.split('_')[0]
            temp_id = f"temp_{prefix}_{cluster_num}"
            
            for ann in anns:
                if 'assigned_id' not in ann:  # Only assign if not already assigned
                    ann['assigned_id'] = temp_id
                ann['final_id'] = temp_id  # Always set final_id
            
            msg = f"Assigned temporary ID {temp_id} to unmatched {prefix} cluster {cluster_num}"
            print(msg)
            f.write(msg + "\n")
        
        completion_msg = "[Stage 7] ID Assignment Completed."
        print(completion_msg)
        f.write(completion_msg + "\n")
        
        # Print final assignment summary
        summary_header = "\nFinal ID Assignment Summary:"
        print(summary_header)
        f.write(summary_header + "\n")
        
        # Print left viewpoint assignments
        left_header = "\nLeft Viewpoint:"
        print(left_header)
        f.write(left_header + "\n")
        
        left_final_grouped = group_annotations_by_LCA_with_viewpoint(data_left, 'left')
        for cl, anns in left_final_grouped.items():
            ids = {ann.get('final_id', 'None') for ann in anns}
            gt_ids = {ann.get('individual_id', 'None') for ann in anns}
            msg = f"Cluster {cl}: GT IDs: {gt_ids} | Assigned IDs: {ids}"
            print(msg)
            f.write(msg + "\n")
        
        # Print right viewpoint assignments
        right_header = "\nRight Viewpoint:"
        print(right_header)
        f.write(right_header + "\n")
        
        right_final_grouped = group_annotations_by_LCA_with_viewpoint(data_right, 'right')
        for cl, anns in right_final_grouped.items():
            ids = {ann.get('final_id', 'None') for ann in anns}
            gt_ids = {ann.get('individual_id', 'None') for ann in anns}
            msg = f"Cluster {cl}: GT IDs: {gt_ids} | Assigned IDs: {ids}"
            print(msg)
            f.write(msg + "\n")
    
    return data_left, data_right

###############################################################################
#                      COMPUTE FINAL METRICS (PREC/REC/F1)                    #
###############################################################################

def compute_final_metrics(data_left_time, data_right_time, data_left_final, data_right_final):
    """
    Enhanced metric calculation with detailed cluster statistics
    """
    print("\n=== Calculating Final Metrics ===")
    
    # First check for mixed individual_id clusters in time-stage data
    print("\nChecking for mixed individual_id clusters in time-stage data:")
    check_mixed_clusters(data_left_time)
    check_mixed_clusters(data_right_time)
    
    # Print cluster counts for time-stage data (ground truth)
    print("\nTime-stage Cluster Counts (Ground Truth):")
    print_cluster_stats(data_left_time, "Left")
    print_cluster_stats(data_right_time, "Right")
    
    # Print cluster counts for final data
    print("\nFinal Cluster Counts:")
    print_cluster_stats(data_left_final, "Left")
    print_cluster_stats(data_right_final, "Right")
    
    # Create mapping of annotation UUID to its original individual_id (from time stage)
    uuid_to_original_id = {}
    for ann in data_left_time["annotations"] + data_right_time["annotations"]:
        uuid_to_original_id[ann["uuid"]] = ann.get("individual_id")
    
    # Create mapping of annotation UUID to its final LCA_clustering_id
    uuid_to_cluster = {}
    for ann in data_left_final["annotations"] + data_right_final["annotations"]:
        uuid_to_cluster[ann["uuid"]] = ann.get("LCA_clustering_id")
    
    # Get all annotation pairs
    all_anns = data_left_final["annotations"] + data_right_final["annotations"]
    n = len(all_anns)
    print(f"\nTotal annotations: {n}")
    print(f"Total possible annotation pairs: {n*(n-1)//2}")
    
    tp, fp, fn = 0, 0, 0
    gt_clusters = defaultdict(set)
    pred_clusters = defaultdict(set)
    
    for i in range(n):
        for j in range(i+1, n):
            ann1 = all_anns[i]
            ann2 = all_anns[j]
            
            # Get cluster assignments from final data
            cluster1 = uuid_to_cluster.get(ann1["uuid"])
            cluster2 = uuid_to_cluster.get(ann2["uuid"])
            
            # Get original ground truth IDs from time stage
            gt1 = uuid_to_original_id.get(ann1["uuid"])
            gt2 = uuid_to_original_id.get(ann2["uuid"])
            
            same_cluster = (cluster1 == cluster2 and cluster1 is not None)
            same_gt = (gt1 == gt2 and gt1 is not None)

            # Track cluster memberships for statistics
            if gt1 is not None:
                gt_clusters[gt1].add(ann1["uuid"])
            if gt2 is not None:
                gt_clusters[gt2].add(ann2["uuid"])
            if cluster1 is not None:
                pred_clusters[cluster1].add(ann1["uuid"])
            if cluster2 is not None:
                pred_clusters[cluster2].add(ann2["uuid"])

            if same_cluster and same_gt:
                tp += 1
            elif same_cluster and not same_gt:
                fp += 1
            elif not same_cluster and same_gt:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) else 0.0
    
    # Calculate additional statistics
    total_gt_individuals = len(gt_clusters)
    total_pred_clusters = len(pred_clusters)
    
    print("\n=== Detailed Metric Calculation ===")
    print(f"True Positives (TP): {tp} - Correctly clustered together")
    print(f"False Positives (FP): {fp} - Incorrectly clustered together")
    print(f"False Negatives (FN): {fn} - Should be clustered but weren't")
    print(f"\nTotal ground truth individuals: {total_gt_individuals}")
    print(f"Total predicted clusters: {total_pred_clusters}")
    
    return {
        "precision": precision, 
        "recall": recall, 
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp, 
        "false_negatives": fn,
        "total_gt_individuals": total_gt_individuals,
        "total_pred_clusters": total_pred_clusters,
        "time_stage_left": data_left_time,
        "time_stage_right": data_right_time
    }

def print_cluster_stats(data, viewpoint):
    """Print detailed cluster statistics for a viewpoint"""
    grouped = group_annotations_by_LCA_all(data)
    print(f"\n{viewpoint} Viewpoint Cluster Statistics:")
    print(f"  Total clusters: {len(grouped)}")
    print(f"  Total annotations: {len(data['annotations'])}")
    
    cluster_sizes = [len(anns) for anns in grouped.values()]
    if cluster_sizes:
        print(f"  Average cluster size: {sum(cluster_sizes)/len(cluster_sizes):.1f}")
        print(f"  Min cluster size: {min(cluster_sizes)}")
        print(f"  Max cluster size: {max(cluster_sizes)}")
    
    # Count individual IDs if present
    if 'individual_id' in data['annotations'][0]:
        individual_ids = set()
        for ann in data['annotations']:
            if ann.get('individual_id'):
                individual_ids.add(ann['individual_id'])
        print(f"  Unique individual IDs: {len(individual_ids)}")

def check_mixed_clusters(data):
    """Check for clusters with mixed individual_ids and print warnings"""
    grouped = group_annotations_by_LCA_all(data)
    for cluster, anns in grouped.items():
        individual_ids = set()
        for ann in anns:
            if 'individual_id' in ann:
                individual_ids.add(ann['individual_id'])
        if len(individual_ids) > 1:
            print(f"WARNING: Cluster {cluster} has mixed individual_ids: {individual_ids}")
            for ann in anns:
                print(f"  Tracking ID: {ann.get('tracking_id')}, Individual ID: {ann.get('individual_id')}, CA_score: {ann.get('CA_score', 0):.4f}")

###############################################################################
#                                  MAIN                                       #
###############################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Post process LCA outputs"
    )
    subparsers = parser.add_subparsers(dest="command")

    old_parser = subparsers.add_parser("old", help="Process given a separate timestamp and metadata file.")
    old_parser.add_argument(
        "images", type=str, help="The image directory."
    )
    old_parser.add_argument(
        "embeddings", type=str, help="The full path to the embeddings file."
    )
    old_parser.add_argument(
        "timestamps", type=str, help="The full path to the timestamp file."
    )
    old_parser.add_argument(
        "metadata", type=str, help="The full path to the metadata (annots) file."
    )
    old_parser.add_argument(
        "merged_annots", type=str, help="The full path to the save the merged file to."
    )
    old_parser.add_argument(
        "in_left", type=str, help="The full path to the left annotations for LCA.",
    )
    old_parser.add_argument(
        "in_right", type=str, help="The full path to the right annotations for LCA.",
    )
    old_parser.add_argument(
        "out_left", type=str, help="The path to save the processed left LCA json."
    )
    old_parser.add_argument(
        "out_right", type=str, help="The path to save the processed right LCA json."
    )

    new_parser = subparsers.add_parser("new", description="Process given a merged csv containing timestamps and metadata information.")  
    new_parser.add_argument(
        "images", type=str, help="The image directory."
    )
    new_parser.add_argument(
        "embeddings", type=str, help="The full path to the embeddings file."
    )
    new_parser.add_argument(
        "merged_annots", type=str, help="The full path to the annotation file, merged with timestamp."
    )
    new_parser.add_argument(
        "in_left", type=str, help="The full path to the left annotations for LCA.",
    )
    new_parser.add_argument(
        "in_right", type=str, help="The full path to the right annotations for LCA.",
    )
    new_parser.add_argument(
        "out_left", type=str, help="The path to save the processed left LCA json."
    )
    new_parser.add_argument(
        "out_right", type=str, help="The path to save the processed right LCA json."
    )

    args = parser.parse_args()

    # Load config first
    with open("algo/config_evaluation_LCA.yaml", "r") as f:
        config = yaml.safe_load(f)

    print(args.command)
    config["old_format"] = args.command == "old"

    # Save args into config
    config["json"]["left"]["input"] = args.in_left
    config["json"]["left"]["output"] = args.out_left
    config["json"]["right"]["input"] = args.in_right
    config["json"]["right"]["output"] = args.out_right
    config["image"]["directory"] = args.images
    config["embedding"]["file"] = args.embeddings
    config["csv"]["merged_bbox"] = args.merged_annots

    if config["old_format"]:
        config["csv"]["metadata"] = args.metadata
        config["csv"]["merged_detections"] = args.timestamps

    # Clear the results file at the start
    results_txt_path = "results_summary.txt"
    with open(results_txt_path, "w") as f:
        f.write("=== Threshold Sweep Results ===\n\n")

# Load embedding settings
    emb_cfg = config.get("embedding", {})
    pickle_file = emb_cfg.get("file", None)
    distance_metric = emb_cfg.get("distance_metric", "euclidean")
    mode = emb_cfg.get("mode", "dual")
    threshold_lower_start = emb_cfg.get("threshold_lower", 0.1)
    threshold_upper_start = emb_cfg.get("threshold_upper", 0.9)  # Default to 0.9 if not specified
    threshold_step = emb_cfg.get("threshold_step", 0.1)

    if mode not in ["dual", "single"]:
        print(f"Invalid mode '{mode}' in config. Use 'dual' or 'single'. Exiting.")
        sys.exit(1)

    if mode == "dual":
        threshold_lower = threshold_lower_start
        threshold_upper = threshold_upper_start
        print(f"Running in DUAL threshold mode with step size {threshold_step}")
    else:  # single mode
        threshold_lower = threshold_lower_start
        threshold_upper = threshold_upper_start  # Use upper as end point
        print(f"Running in SINGLE threshold mode, sweeping from {threshold_lower} to {threshold_upper} with step {threshold_step}")

    while True:
        if mode == "dual" and (threshold_lower > 0.5 or threshold_upper < 0.5):
            break
        elif mode == "single" and threshold_lower > threshold_upper:
            break

        if mode == "single":
            print(f"\n{'#'*80}")
            print(f"### RUNNING WITH SINGLE THRESHOLD: {threshold_lower:.2f} ###")
        else:
            print(f"\n{'#'*80}")
            print(f"### RUNNING WITH THRESHOLDS: lower={threshold_lower:.2f}, upper={threshold_upper:.2f} ###")
        print(f"{'#'*80}\n")

        # Reset global decision log
        global GLOBAL_DECISION_LOG
        GLOBAL_DECISION_LOG = {
            "split_auto_merge_count": 0,
            "split_auto_no_merge_count": 0,
            "split_ambiguous_count": 0,
            "time_auto_merge_count": 0,
            "time_auto_no_merge_count": 0,
            "time_ambiguous_count": 0,
        }

        if config['old_format']:
            # =========== Stage 2 ===========
            print("\n--- Stage 2: Merging CSVs with Bounding Boxes ---")
            merge_csvs_with_bbox(
                config['csv']['merged_detections'],
                config['csv']['metadata'],
                config['csv']['merged_bbox']
            )

        # =========== Stage 3 ===========
        print("\n--- Stage 3: Updating JSON Annotations with Timestamps ---")
        update_json_with_timestamp(
            config['json']['left']['input'],
            config['csv']['merged_bbox'],
            config['json']['left']['output']
        )
        update_json_with_timestamp(
            config['json']['right']['input'],
            config['csv']['merged_bbox'],
            config['json']['right']['output']
        )

        # Load updated JSON
        with open(config['json']['left']['output'], "r") as f:
            data_left = json.load(f)
        with open(config['json']['right']['output'], "r") as f:
            data_right = json.load(f)

        print("\nInitial Left Viewpoint Cluster Summary:")
        print_cluster_summary(data_left)
        print_viewpoint_cluster_mapping(data_left, "left")

        print("\nInitial Right Viewpoint Cluster Summary:")
        print_cluster_summary(data_right)
        print_viewpoint_cluster_mapping(data_right, "right")

        # =========== EMBEDDING LOADING & SETTINGS ===========
        if not pickle_file or not os.path.exists(pickle_file):
            print("\n[Warning] Embedding pickle file not found or not specified; merges will skip.")
            embedding_matrix, uuid_to_index = [], {}
        else:
            embedding_matrix, uuid_to_index = load_embeddings(pickle_file)

        # =========== Stage 4: Split Cluster Verification ===========
        print("\n--- Stage 4: Split Cluster Verification (Embedding-based) ---")
        grouped_left = group_annotations_by_LCA_all(data_left)
        grouped_right = group_annotations_by_LCA_all(data_right)

        iteration = 0
        while True:
            iteration += 1
            print(f"\n-- Split Cluster Verification Iteration: {iteration} --")
            changed = consistency_check_with_embeddings(
                grouped_left, grouped_right,
                data_left, data_right,
                embedding_matrix, uuid_to_index,
                threshold_lower, threshold_upper if mode == "dual" else threshold_lower,
                distance_metric,
                stage_name="split"
            )
            if not changed:
                print("No further merges in split stage. Proceeding.")
                break
            grouped_left = group_annotations_by_LCA_all(data_left)
            grouped_right = group_annotations_by_LCA_all(data_right)

        # Save JSON after split
        data_left["annotations"] = []
        for cl, anns in grouped_left.items():
            data_left["annotations"].extend(anns)
        data_right["annotations"] = []
        for cr, anns in grouped_right.items():
            data_right["annotations"].extend(anns)

        suffix = f"_{threshold_lower:.1f}_{threshold_upper:.1f}" if mode == "dual" else f"_{threshold_lower:.1f}"
        left_split_output = config['json']['left']['output'].replace(".json", f"_split{suffix}.json")
        right_split_output = config['json']['right']['output'].replace(".json", f"_split{suffix}.json")

        with open(left_split_output, "w") as f:
            json.dump(data_left, f, indent=4)
        with open(right_split_output, "w") as f:
            json.dump(data_right, f, indent=4)
        print(f"[Split Stage] Saved updated left JSON to: {left_split_output}")
        print(f"[Split Stage] Saved updated right JSON to: {right_split_output}")

        # =========== Stage 6: Time-Overlap ===========
        print("\n--- Stage 6: Time-Overlap Verification (Embedding-based) ---")
        grouped_left = group_annotations_by_LCA_all(data_left)
        grouped_right = group_annotations_by_LCA_all(data_right)

        print("\nLeft viewpoint time-overlap check:")
        time_overlap_verification_with_embeddings(
            grouped_left,
            {"distance_metric": distance_metric},
            embedding_matrix, uuid_to_index,
            threshold_lower, threshold_upper if mode == "dual" else threshold_lower,
            distance_metric
        )

        print("\nRight viewpoint time-overlap check:")
        time_overlap_verification_with_embeddings(
            grouped_right,
            {"distance_metric": distance_metric},
            embedding_matrix, uuid_to_index,
            threshold_lower, threshold_upper if mode == "dual" else threshold_lower,
            distance_metric
        )

        # re-save after time
        data_left["annotations"] = []
        for cl, anns in grouped_left.items():
            data_left["annotations"].extend(anns)
        data_right["annotations"] = []
        for cr, anns in grouped_right.items():
            data_right["annotations"].extend(anns)

        left_time_output = config['json']['left']['output'].replace(".json", f"_time{suffix}.json")
        right_time_output = config['json']['right']['output'].replace(".json", f"_time{suffix}.json")

        with open(left_time_output, "w") as f:
            json.dump(data_left, f, indent=4)
        with open(right_time_output, "w") as f:
            json.dump(data_right, f, indent=4)
        print(f"[Time Stage] Saved updated left JSON to: {left_time_output}")
        print(f"[Time Stage] Saved updated right JSON to: {right_time_output}")

        # Save time-stage data BEFORE ID assignment
        time_stage_left = json.loads(json.dumps(data_left))  # Deep copy
        time_stage_right = json.loads(json.dumps(data_right))  # Deep copy

        # =========== Stage 5 & 7: Equivalence and ID Assignment ===========
        grouped_left_wv = group_annotations_by_LCA_with_viewpoint(data_left, 'left')
        grouped_right_wv = group_annotations_by_LCA_with_viewpoint(data_right, 'right')
        check_cluster_equivalence(grouped_left_wv, grouped_right_wv)
        assign_ids_after_equivalence_check(data_left, data_right, output_file=results_txt_path)

        # Final Save
        final_left = config['json']['left']['output'].replace(".json", f"_final{suffix}.json")
        final_right = config['json']['right']['output'].replace(".json", f"_final{suffix}.json")

        with open(final_left, "w") as f:
            json.dump(data_left, f, indent=4)
        with open(final_right, "w") as f:
            json.dump(data_right, f, indent=4)

        # Compute metrics
        metrics = compute_final_metrics(
            time_stage_left, time_stage_right,
            data_left, data_right
        )

        print(f"\n=== Final Metrics ===")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")

        # Write results
        write_results_to_txt(results_txt_path, metrics, GLOBAL_DECISION_LOG, config, threshold_lower, threshold_upper if mode == "dual" else None)

        # Compute additional metrics
        all_final_ids = set()
        for ann in data_left["annotations"] + data_right["annotations"]:
            fid = ann.get("final_id", None)
            if fid is not None and fid != "None":
                all_final_ids.add(fid)
        assigned_id_count = len(all_final_ids)

        manual_count = (GLOBAL_DECISION_LOG["split_ambiguous_count"] +
                        GLOBAL_DECISION_LOG["time_ambiguous_count"])
        auto_count = (GLOBAL_DECISION_LOG["split_auto_merge_count"] +
                      GLOBAL_DECISION_LOG["split_auto_no_merge_count"] +
                      GLOBAL_DECISION_LOG["time_auto_merge_count"] +
                      GLOBAL_DECISION_LOG["time_auto_no_merge_count"])
        
        ground_truth_count = metrics["total_gt_individuals"]
        accuracy = assigned_id_count / ground_truth_count if ground_truth_count else 0.0

        threshold_str = f"({threshold_lower:.1f},{threshold_upper:.1f})" if mode == "dual" else f"({threshold_lower:.1f})"
        GLOBAL_RUN_RESULTS.append({
            "threshold": threshold_str,
            "manual_decisions": manual_count,
            "automatic_decisions": auto_count,
            "assigned_ids": assigned_id_count,
            "ground_truth_ids": ground_truth_count,
            "accuracy": accuracy,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        })

        # Update thresholds for next iteration in dual mode
        if mode == "dual":
            threshold_lower = round(threshold_lower + threshold_step, 2)
            threshold_upper = round(threshold_upper - threshold_step, 2)
        else:  # single mode
            threshold_lower = round(threshold_lower + threshold_step, 2)
    # Final summary and plotting (unchanged)
    print("\nAll threshold iterations completed.")
    with open(results_txt_path, "a") as f:
        f.write("\n=== Final Threshold Results Summary ===\n")
        f.write("threshold,manual_decisions,automatic_decisions,assigned_ids,ground_truth_ids,accuracy\n")
        for row in GLOBAL_RUN_RESULTS:
            f.write(f"{row['threshold']},"
                    f"{row['manual_decisions']},"
                    f"{row['automatic_decisions']},"
                    f"{row['assigned_ids']},"
                    f"{row['ground_truth_ids']},"
                    f"{row['accuracy']:.3f}\n")

    print("\n=== Final Threshold Results Summary ===")
    print("threshold,manual_decisions,automatic_decisions,assigned_ids,ground_truth_ids,accuracy")
    for row in GLOBAL_RUN_RESULTS:
        print(f"{row['threshold']},"
              f"{row['manual_decisions']},"
              f"{row['automatic_decisions']},"
              f"{row['assigned_ids']},"
              f"{row['ground_truth_ids']},"
              f"{row['accuracy']:.3f}")

    # Plotting (unchanged)
    manual_list = [r["manual_decisions"] for r in GLOBAL_RUN_RESULTS]
    accuracy_list = [r["accuracy"] for r in GLOBAL_RUN_RESULTS]
    plt.figure()
    plt.plot(manual_list, accuracy_list, marker='o', label='Manual vs. Accuracy')
    plt.xlabel("Number of Manual Decisions")
    plt.ylabel("Accuracy (Unique Final IDs / Ground Truth IDs)")
    plt.title("Manual Decisions vs. ID Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("manual_vs_accuracy.png", dpi=150)
    plt.show()

    # Precision/Recall/F1 plot
    precisions = [r["precision"] for r in GLOBAL_RUN_RESULTS]
    recalls = [r["recall"] for r in GLOBAL_RUN_RESULTS]
    f1s = [r["f1"] for r in GLOBAL_RUN_RESULTS]
    plt.figure()
    plt.plot(manual_list, precisions, marker='o', label='Precision')
    plt.plot(manual_list, recalls, marker='o', label='Recall')
    plt.plot(manual_list, f1s, marker='o', label='F1')
    plt.xlabel("Manual Decisions")
    plt.ylabel("Metric Value")
    plt.title("Precision / Recall / F1 vs. Manual Decisions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("precision_recall_f1_vs_manual.png", dpi=150)
    plt.show()

# Update pairwise_verification_with_embedding to handle single mode
def pairwise_verification_with_embedding(
    grouped_annotations, cluster1, cluster2, data_dict,
    embedding_matrix, uuid_to_index,
    threshold_lower, threshold_upper,
    stage
):
    if cluster1 not in grouped_annotations or cluster2 not in grouped_annotations:
        return False
    if not grouped_annotations[cluster1] or not grouped_annotations[cluster2]:
        return False

    emb1, best_ann1 = get_cluster_best_annotation_embedding(grouped_annotations[cluster1], embedding_matrix, uuid_to_index)
    emb2, best_ann2 = get_cluster_best_annotation_embedding(grouped_annotations[cluster2], embedding_matrix, uuid_to_index)

    if emb1 is None or emb2 is None:
        print(f"[{stage.capitalize()}] WARNING: Missing embedding for {cluster1} or {cluster2}. Skipping merge.")
        return False

    metric = data_dict.get("distance_metric", "euclidean")
    if metric == "cosine":
        dist = bounded_cosine_distance(emb1, emb2)
    else:
        dist = euclidean_distance(emb1, emb2)

    print(f"[{stage.capitalize()}] Checking {cluster1} vs {cluster2}, distance={dist:.4f} (metric={metric})")

    if threshold_upper is None:  # Single threshold mode
        if dist < threshold_lower:
            print(f"[{stage.capitalize()}] => distance < {threshold_lower}, auto merge.")
            update_cluster_merge(grouped_annotations, cluster2, cluster1)
            if stage == "split":
                GLOBAL_DECISION_LOG["split_auto_merge_count"] += 1
            else:
                GLOBAL_DECISION_LOG["time_auto_merge_count"] += 1
            return True
        else:
            print(f"[{stage.capitalize()}] => distance >= {threshold_lower}, auto no-merge.")
            if stage == "split":
                GLOBAL_DECISION_LOG["split_auto_no_merge_count"] += 1
            else:
                GLOBAL_DECISION_LOG["time_auto_no_merge_count"] += 1
            return False
    else:  # Dual threshold mode (original logic)
        if dist < threshold_lower:
            print(f"[{stage.capitalize()}] => distance < {threshold_lower}, auto merge.")
            update_cluster_merge(grouped_annotations, cluster2, cluster1)
            if stage == "split":
                GLOBAL_DECISION_LOG["split_auto_merge_count"] += 1
            else:
                GLOBAL_DECISION_LOG["time_auto_merge_count"] += 1
            return True
        elif dist > threshold_upper:
            print(f"[{stage.capitalize()}] => distance > {threshold_upper}, auto no-merge.")
            if stage == "split":
                GLOBAL_DECISION_LOG["split_auto_no_merge_count"] += 1
            else:
                GLOBAL_DECISION_LOG["time_auto_no_merge_count"] += 1
            return False
        else:
            same_gt = (best_ann1.get("individual_id") == best_ann2.get("individual_id"))
            if same_gt:
                print(f"[{stage.capitalize()}] => ambiguous but same GT => merge.")
                update_cluster_merge(grouped_annotations, cluster2, cluster1)
                if stage == "split":
                    GLOBAL_DECISION_LOG["split_ambiguous_count"] += 1
                else:
                    GLOBAL_DECISION_LOG["time_ambiguous_count"] += 1
                return True
            else:
                print(f"[{stage.capitalize()}] => ambiguous but different GT => no merge.")
                if stage == "split":
                    GLOBAL_DECISION_LOG["split_ambiguous_count"] += 1
                else:
                    GLOBAL_DECISION_LOG["time_ambiguous_count"] += 1
                return False

if __name__ == "__main__":
    main()