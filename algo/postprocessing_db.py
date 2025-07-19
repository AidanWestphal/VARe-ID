#!/usr/bin/env python3
import os
import sys
import time
import yaml
import pandas as pd
import csv
import re
import ast
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path to import from UI folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import database functions
from UI.db_scripts import init_db, add_image_pair, get_decision, get_instance_stats, get_existing_pair_decision, check_pair_exists

# Store metadata for each pair (not needed in database)
pair_metadata = {}

# -------------------------
# Helper Functions
# -------------------------
def wait_for_database_decisions(db_path, pair_ids, check_interval=5):
    """Wait for specific pairs to be completed"""
    if not pair_ids:
        return
    
    print(f"\nWaiting for {len(pair_ids)} verification decisions...")
    print("Please complete verification tasks in the UI...")
    
    while True:
        # Check status of specific pairs
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        placeholders = ','.join('?' * len(pair_ids))
        cursor.execute(f"""
            SELECT COUNT(*) FROM image_verification
            WHERE id IN ({placeholders}) AND status IN ('awaiting', 'in_progress')
        """, pair_ids)
        pending = cursor.fetchone()[0]
        conn.close()
        
        if pending == 0:
            break
            
        print(f"Pending tasks: {pending}/{len(pair_ids)} - Checking again in {check_interval} seconds...")
        time.sleep(check_interval)
    
    print(f"All verification tasks completed!")


def save_json_with_stage(data, original_filename, stage_suffix):
    base, ext = os.path.splitext(original_filename)
    new_filename = base + "_" + stage_suffix + ext
    with open(new_filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved file: {new_filename}")
    return new_filename


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


# =========================
# Stage 1: Merge Detection CSV Files
# =========================
def merge_csv_files(file1_path, file2_path, output_path):
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    merged_df = pd.concat([df1, df2], ignore_index=True)
    merged_df = merged_df.sort_values("timestamp")
    merged_df.to_csv(output_path, index=False)
    print(f"[Stage 1] Merged detection CSV files into: {output_path}")
    print(f"[Stage 1] Total rows: {len(merged_df)}")


# =========================
# Stage 2: Merge CSVs with Bounding Boxes
# =========================
def parse_bbox_string_xyxy(bbox_str):
    s = bbox_str.strip()
    s = re.sub(r"[\[\]\(\)]", "", s)
    parts = s.split(",")
    try:
        return tuple(int(p.strip()) for p in parts)
    except Exception as e:
        print(f"Error parsing bbox (xyxy): {bbox_str} -> {e}")
        return None


def parse_bbox_string_xywh(bbox_str):
    s = bbox_str.strip()
    s = re.sub(r"[\[\]\(\)]", "", s)
    parts = s.split(",")
    try:
        coords = [int(p.strip()) for p in parts]
        if len(coords) != 4:
            return None
        x1, y1, w, h = coords
        return (x1, y1, x1 + w, y1 + h)
    except Exception as e:
        print(f"Error parsing bbox (xywh): {bbox_str} -> {e}")
        return None


def merge_csvs_with_bbox(timestamps_csv, metadata_csv, output_csv):
    dict_ts = {}
    with open(timestamps_csv, "r", newline="", encoding="utf-8") as f1:
        reader1 = csv.DictReader(f1)
        ts_fieldnames = reader1.fieldnames or []
        for row in reader1:
            ts_filename = row.get("frame_name")
            if not ts_filename:
                continue
            ts_bbox_str = row.get("bounding_box")
            if not ts_bbox_str:
                continue
            ts_bbox = parse_bbox_string_xyxy(ts_bbox_str)
            if ts_bbox is None:
                continue
            key = (ts_filename, ts_bbox)
            dict_ts[key] = row

    merged_rows = []
    combined_fieldnames = set(ts_fieldnames)

    with open(metadata_csv, "r", newline="", encoding="utf-8") as f2:
        reader2 = csv.DictReader(f2)
        metadata_fieldnames = reader2.fieldnames or []
        for col in metadata_fieldnames:
            combined_fieldnames.add(col)
        for row in reader2:
            md_filename = row.get("file_name")
            if not md_filename:
                continue
            md_bbox_str = row.get("bbox")
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
    with open(output_csv, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=combined_fieldnames)
        writer.writeheader()
        for row in merged_rows:
            writer.writerow(row)

    print(f"[Stage 2] Merged CSV with bounding boxes written to: {output_csv}")
    print(f"[Stage 2] Total merged rows: {len(merged_rows)}")


# =========================
# Stage 3: Update JSON Annotations with Timestamps
# =========================
def parse_bbox(bbox_str):
    try:
        return list(ast.literal_eval(bbox_str))
    except Exception as e:
        print(f"Error parsing bbox from CSV: {bbox_str} -> {e}")
        return None


def make_comparable_dict_from_csv(row, file_to_uuid):
    """Create a dictionary of only the fields used for matching JSON annotations."""
    try:
        tracking_id = (
            int(row["tracking_id"].strip()) if row["tracking_id"].strip() else None
        )
    except:
        tracking_id = None
    try:
        confidence = (
            float(row["confidence"].strip()) if row["confidence"].strip() else None
        )
    except:
        confidence = None
    try:
        detection_class = (
            int(row["detection_class"].strip())
            if row["detection_class"].strip()
            else None
        )
    except:
        detection_class = None
    try:
        CA_score = float(row["CA_score"].strip()) if row["CA_score"].strip() else None
    except:
        CA_score = None

    species = row["species"].strip() if "species" in row else None
    bbox = parse_bbox(row["bbox"])
    image_uuid = file_to_uuid.get(row["file_name"].strip(), None)
    timestamp = row["timestamp"].strip()

    comparable_fields = {
        "image_uuid": image_uuid,
        "tracking_id": tracking_id,
        "confidence": confidence,
        "detection_class": detection_class,
        "species": species,
        "bbox": bbox,
        "CA_score": CA_score,
    }
    return comparable_fields, timestamp


def make_comparable_dict_from_json(ann):
    """Create a dictionary of only the fields used for matching to CSV."""
    bbox_list = list(ann["bbox"]) if ann.get("bbox") else None
    return {
        "image_uuid": ann["image_uuid"],
        "tracking_id": ann["tracking_id"],
        "confidence": ann["confidence"],
        "detection_class": ann["detection_class"],
        "species": ann["species"],
        "bbox": bbox_list,
        "CA_score": ann["CA_score"],
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

    updated_annotations = []
    for csv_fields, csv_timestamp in csv_common_list:
        for ann in data["annotations"]:
            ann_fields = make_comparable_dict_from_json(ann)
            if ann_fields == csv_fields:
                ann["timestamp"] = csv_timestamp
                updated_annotations.append(
                    {
                        "annotation_uuid": ann["uuid"],
                        "image_uuid": ann["image_uuid"],
                        "tracking_id": ann["tracking_id"],
                        "timestamp_added": csv_timestamp,
                    }
                )
                break

    with open(json_output, "w") as f:
        json.dump(data, f, indent=4)

    print(f"[Stage 3] Updated JSON written to: {json_output}")
    print(f"[Stage 3] Number of annotations updated: {len(updated_annotations)}")


# =========================
# Stage 4: Split Cluster Verification (Database-driven)
# =========================
def group_annotations_by_LCA_all(data):
    annotations = data["annotations"]
    grouped = defaultdict(list)
    for ann in annotations:
        lca = ann.get("LCA_clustering_id")
        if lca is not None:
            grouped[lca].append(ann)
    return grouped


def get_image_filename(images, image_uuid, image_dir):
    if image_dir is None:
        return None
    for img in images:
        if img["uuid"] == image_uuid:
            return os.path.join(image_dir, img["file_name"])
    return None


def prepare_verification_task(grouped_annotations, cluster1, cluster2, data, image_dir, db_path, viewpoint, stage):
    """Prepare and submit a verification task to the database. Returns pair ID if needs decision, None otherwise."""
    global pair_metadata
    
    if cluster1 not in grouped_annotations or cluster2 not in grouped_annotations:
        return None

    if not grouped_annotations[cluster1] or not grouped_annotations[cluster2]:
        return None

    best_ann1 = max(grouped_annotations[cluster1], key=lambda ann: ann["CA_score"])
    best_ann2 = max(grouped_annotations[cluster2], key=lambda ann: ann["CA_score"])

    # Order UUIDs to create consistent ID
    uuid1 = best_ann1['uuid']
    uuid2 = best_ann2['uuid']
    if uuid1 > uuid2:
        uuid1, uuid2 = uuid2, uuid1
    
    # Create unique ID with ordered UUIDs
    unique_id = f"{uuid1}_{uuid2}_{stage}_{viewpoint}"

    # Check if this pair already exists in the database
    exists, status, decision = check_pair_exists(best_ann1['uuid'], best_ann2['uuid'], db_path)
    
    if exists:
        if status in ['checked', 'sent'] and decision != 'none':
            # Pair already verified, use existing decision
            print(f"Pair already verified: {best_ann1['uuid']} - {best_ann2['uuid']} = {decision}")
            
            # Map decision and apply it immediately
            merge = (decision == 'correct')
            
            if stage == "split":
                update_split_cluster(data, grouped_annotations, cluster1, cluster2, merge=merge)
            elif stage == "time":
                update_time_overlap(data, grouped_annotations, cluster1, cluster2, merge=merge)
            
            return None  # No decision needed
        else:
            # Pair exists but not yet decided
            print(f"Pair already in database (status: {status}): {best_ann1['uuid']} - {best_ann2['uuid']}")
            return unique_id  # Need to wait for decision

    # Get image paths (will be None if image_dir is None)
    image_path1 = get_image_filename(data["images"], best_ann1["image_uuid"], image_dir)
    image_path2 = get_image_filename(data["images"], best_ann2["image_uuid"], image_dir)

    # Extract bounding boxes and convert from [x,y,w,h] to [x1,y1,x2,y2]
    bbox1 = None
    bbox2 = None
    
    if "bbox" in best_ann1 and best_ann1["bbox"]:
        x, y, w, h = best_ann1["bbox"]
        bbox1 = json.dumps([x, y, x + w, y + h])
    
    if "bbox" in best_ann2 and best_ann2["bbox"]:
        x, y, w, h = best_ann2["bbox"]
        bbox2 = json.dumps([x, y, x + w, y + h])

    # Re-order all data to ensure uuid1 < uuid2
    if best_ann1['uuid'] > best_ann2['uuid']:
        # Swap everything
        image_path1, image_path2 = image_path2, image_path1
        bbox1, bbox2 = bbox2, bbox1
        best_ann1, best_ann2 = best_ann2, best_ann1
    
    # Store metadata
    pair_metadata[unique_id] = {
        'viewpoint': viewpoint,
        'stage': stage,
        'ca_score1': best_ann1['CA_score'],
        'ca_score2': best_ann2['CA_score']
    }
    
    # Add to database
    add_image_pair(
        unique_id,
        uuid1,
        image_path1 or "NO_IMAGE",
        bbox1,
        uuid2,
        image_path2 or "NO_IMAGE",
        bbox2,
        db_path
    )
    
    return unique_id  # Need to wait for decision


def process_verification_decision(grouped_annotations, cluster1, cluster2, data, db_path, viewpoint, stage):
    """Process a verification decision from the database"""
    unique_id = f"{cluster1}_{cluster2}_{stage}_{viewpoint}"
    decision = get_decision(unique_id, db_path)
    
    if decision is None:
        return None
    
    # Map decisions: 'correct' -> merge, 'incorrect' -> no_merge
    merge = (decision == 'correct')
    
    if stage == "split":
        update_split_cluster(data, grouped_annotations, cluster1, cluster2, merge=merge)
    elif stage == "time":
        update_time_overlap(data, grouped_annotations, cluster1, cluster2, merge=merge)
    
    return merge


def update_split_cluster(data, grouped_annotations, cluster1, cluster2, merge=True):
    """
    If merging, move cluster1's annotations into cluster2, then delete cluster1.
    Otherwise, update the tracking IDs if there's overlap.
    """
    if merge:
        # Merge cluster1 into cluster2
        grouped_annotations[cluster2].extend(grouped_annotations[cluster1])
        for ann in grouped_annotations[cluster1]:
            ann["LCA_clustering_id"] = cluster2
        del grouped_annotations[cluster1]
        print(f"[Split Stage] Clusters {cluster1} and {cluster2} merged into {cluster2}.")
    else:
        ids1 = set(ann["tracking_id"] for ann in grouped_annotations[cluster1])
        ids2 = set(ann["tracking_id"] for ann in grouped_annotations[cluster2])
        common_ids = ids1.intersection(ids2)
        if common_ids:
            for ann in grouped_annotations[cluster2]:
                if ann["tracking_id"] in common_ids:
                    ann["tracking_id"] = str(ann["tracking_id"]) + "_new"
            print(f"[Split Stage] Updated common tracking IDs {common_ids} in cluster {cluster2} to include '_new'.")


def update_time_overlap(data, grouped_annotations, cluster1, cluster2, merge=True):
    """If merging, move cluster2's annotations to cluster1. Otherwise, do nothing."""
    if merge:
        for ann in grouped_annotations[cluster2]:
            ann["LCA_clustering_id"] = cluster1
        grouped_annotations[cluster1].extend(grouped_annotations[cluster2])
        del grouped_annotations[cluster2]
        print(f"[Time Stage] Clusters {cluster1} and {cluster2} merged into {cluster1}.")
    else:
        print(f"[Time Stage] No merging performed for clusters {cluster1} and {cluster2}.")


def consistency_check_with_database(grouped_left, grouped_right, data_left, data_right, image_dir, db_path):
    """Submit all split cluster verification tasks to database and return list of pair IDs needing decisions"""
    pair_ids_needing_decisions = []
    
    # Left viewpoint
    left_tracking = defaultdict(set)
    for lca, anns in grouped_left.items():
        for ann in anns:
            left_tracking[ann["tracking_id"]].add(lca)
    
    for tid, clusters in left_tracking.items():
        if len(clusters) > 1:
            print(f"\n[Left VP] Tracking ID {tid} has split clusters: {clusters}")
            cl_list = list(clusters)
            
            for i in range(len(cl_list) - 1):
                c1 = cl_list[i]
                c2 = cl_list[i + 1]
                pair_id = prepare_verification_task(grouped_left, c1, c2, data_left, image_dir, db_path, "left", "split")
                if pair_id:
                    pair_ids_needing_decisions.append(pair_id)
    
    # Right viewpoint
    right_tracking = defaultdict(set)
    for lca, anns in grouped_right.items():
        for ann in anns:
            right_tracking[ann["tracking_id"]].add(lca)
    
    for tid, clusters in right_tracking.items():
        if len(clusters) > 1:
            print(f"\n[Right VP] Tracking ID {tid} has split clusters: {clusters}")
            cl_list = list(clusters)
            
            for i in range(len(cl_list) - 1):
                c1 = cl_list[i]
                c2 = cl_list[i + 1]
                pair_id = prepare_verification_task(grouped_right, c1, c2, data_right, image_dir, db_path, "right", "split")
                if pair_id:
                    pair_ids_needing_decisions.append(pair_id)
    
    if pair_ids_needing_decisions:
        print(f"\nTotal pairs needing decisions: {len(pair_ids_needing_decisions)}")
    
    return pair_ids_needing_decisions


def process_split_decisions(grouped_left, grouped_right, data_left, data_right, db_path):
    """Process all split cluster decisions from database"""
    # Process left viewpoint decisions
    left_tracking = defaultdict(set)
    for lca, anns in grouped_left.items():
        for ann in anns:
            left_tracking[ann["tracking_id"]].add(lca)
    
    for tid, clusters in left_tracking.items():
        if len(clusters) > 1:
            cl_list = list(clusters)
            for i in range(len(cl_list) - 1):
                c1 = cl_list[i]
                c2 = cl_list[i + 1]
                if c1 in grouped_left and c2 in grouped_left:
                    process_verification_decision(grouped_left, c1, c2, data_left, db_path, "left", "split")
    
    # Process right viewpoint decisions
    right_tracking = defaultdict(set)
    for lca, anns in grouped_right.items():
        for ann in anns:
            right_tracking[ann["tracking_id"]].add(lca)
    
    for tid, clusters in right_tracking.items():
        if len(clusters) > 1:
            cl_list = list(clusters)
            for i in range(len(cl_list) - 1):
                c1 = cl_list[i]
                c2 = cl_list[i + 1]
                if c1 in grouped_right and c2 in grouped_right:
                    process_verification_decision(grouped_right, c1, c2, data_right, db_path, "right", "split")


# =========================
# Stage 6: Time-Overlap Verification (Database-driven)
# =========================
def parse_timestamp(ts_str):
    """Parse a timestamp string of the form 'YYYY-MM-DD HH:MM:SS,ffffff' into a datetime."""
    date_part, time_part = ts_str.split(" ")
    first_comma = time_part.find(",")
    if first_comma != -1:
        time_part = time_part[:first_comma] + "." + time_part[first_comma + 1 :]
        time_part = time_part.replace(",", "")
    fixed = f"{date_part} {time_part}"
    return datetime.strptime(fixed, "%Y-%m-%d %H:%M:%S.%f")


def intervals_overlap(start1, end1, start2, end2):
    return start1 <= end2 and start2 <= end1


def find_clusters_with_no_time_overlap(grouped_annotations, threshold=timedelta(seconds=1)):
    """Find pairs of clusters in the same viewpoint that have zero overlapping times."""
    cluster_intervals = defaultdict(dict)
    for cl, anns in grouped_annotations.items():
        tid_to_times = defaultdict(list)
        for ann in anns:
            if "timestamp" not in ann:
                continue
            ts_str = ann["timestamp"]
            try:
                dt = parse_timestamp(ts_str)
                tid_to_times[ann["tracking_id"]].append(dt)
            except:
                continue
        for tid, times in tid_to_times.items():
            times.sort()
            cluster_intervals[cl][tid] = (times[0], times[-1])

    clusters = sorted(cluster_intervals.keys())
    no_overlap = []
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            c1 = clusters[i]
            c2 = clusters[j]
            found = False
            for t1 in cluster_intervals[c1]:
                start1, end1 = cluster_intervals[c1][t1]
                for t2 in cluster_intervals[c2]:
                    start2, end2 = cluster_intervals[c2][t2]
                    if intervals_overlap(start1, end1, start2, end2):
                        found = True
                        break
                if found:
                    break
            if not found:
                no_overlap.append((c1, c2))
    return no_overlap


def time_overlap_verification_db(grouped_annotations, data, image_dir, db_path, viewpoint):
    """Submit time overlap verification tasks to database and return list of pair IDs needing decisions"""
    print(f"\n[Stage 6] Checking for cluster pairs with NO time overlap in {viewpoint} viewpoint...")
    no_overlap_pairs = find_clusters_with_no_time_overlap(grouped_annotations)
    
    if not no_overlap_pairs:
        print("All clusters have some time overlap.")
        return []
    
    print(f"Found {len(no_overlap_pairs)} cluster pair(s) with no time overlap.")
    pair_ids_needing_decisions = []
    
    for c1, c2 in no_overlap_pairs:
        print(f"\nCluster pair with NO overlap: {c1} & {c2}")
        pair_id = prepare_verification_task(grouped_annotations, c1, c2, data, image_dir, db_path, viewpoint, "time")
        if pair_id:
            pair_ids_needing_decisions.append(pair_id)
    
    if pair_ids_needing_decisions:
        print(f"Pairs needing decisions: {len(pair_ids_needing_decisions)}")
    
    return pair_ids_needing_decisions


def process_time_decisions(grouped_annotations, data, db_path, viewpoint):
    """Process time overlap decisions from database"""
    no_overlap_pairs = find_clusters_with_no_time_overlap(grouped_annotations)
    
    for c1, c2 in no_overlap_pairs:
        if c1 in grouped_annotations and c2 in grouped_annotations:
            process_verification_decision(grouped_annotations, c1, c2, data, db_path, viewpoint, "time")


# =========================
# Stage 5: Cluster Equivalence
# =========================
def group_annotations_by_LCA_with_viewpoint(data, viewpoint):
    grouped = defaultdict(list)
    for ann in data["annotations"]:
        lca = ann.get("LCA_clustering_id")
        if lca is not None:
            grouped[f"{lca}_{viewpoint}"].append(ann)
    return grouped


def check_cluster_equivalence(grouped_left, grouped_right):
    print("\nStage 5: Checking Cluster Equivalence")
    print("=====================================\n")

    tracking_left = {}
    tracking_right = {}

    for cl, anns in grouped_left.items():
        for ann in anns:
            tracking_left[ann["tracking_id"]] = cl

    for cl, anns in grouped_right.items():
        for ann in anns:
            tracking_right[ann["tracking_id"]] = cl

    equiv = {"left_to_right": defaultdict(set), "right_to_left": defaultdict(set)}

    for left_cl, anns in grouped_left.items():
        left_tids = {ann["tracking_id"] for ann in anns}
        rcs = {tracking_right[tid] for tid in left_tids if tid in tracking_right}
        equiv["left_to_right"][left_cl] = rcs

    for right_cl, anns in grouped_right.items():
        right_tids = {ann["tracking_id"] for ann in anns}
        lcs = {tracking_left[tid] for tid in right_tids if tid in tracking_left}
        equiv["right_to_left"][right_cl] = lcs

    return equiv


# =========================
# Stage 7: ID Assignment
# =========================
def assign_ids_after_equivalence_check(data_left, data_right):
    grouped_left = group_annotations_by_LCA_with_viewpoint(data_left, "left")
    grouped_right = group_annotations_by_LCA_with_viewpoint(data_right, "right")
    equiv = check_cluster_equivalence(grouped_left, grouped_right)

    individual_id = 1
    processed = set()
    
    # Process equivalences and assign IDs
    for left_cl, right_cls in equiv["left_to_right"].items():
        if right_cls and left_cl not in processed:
            # Assign same ID to equivalent clusters
            for ann in grouped_left[left_cl]:
                ann["individual_id"] = individual_id
            processed.add(left_cl)
            
            for right_cl in right_cls:
                if right_cl not in processed:
                    for ann in grouped_right[right_cl]:
                        ann["individual_id"] = individual_id
                    processed.add(right_cl)
            
            individual_id += 1
    
    # Handle unmatched clusters
    for cl, anns in grouped_left.items():
        if cl not in processed:
            for ann in anns:
                ann["individual_id"] = f"temp_left_{cl}"
    
    for cl, anns in grouped_right.items():
        if cl not in processed:
            for ann in anns:
                ann["individual_id"] = f"temp_right_{cl}"
    
    print("[Stage 7] ID Assignment Completed.")


# =========================
# Main Workflow
# =========================
def main():
    with open("./postprocessing_db.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Get database path and image directory
    db_path = config.get("database", {}).get("path", "./cluster_verification.db")
    image_dir = config.get("image", {}).get("directory")
    
    # Initialize database
    init_db(db_path)
    
    if image_dir is None:
        print("\n[INFO] Image directory is set to None - verification will proceed without images")

    # Stage 1: Merge detection CSV files
    print("\n--- Stage 1: Merging Detection CSV Files ---")
    merge_csv_files(
        config["csv"]["detections1"],
        config["csv"]["detections2"],
        config["csv"]["merged_detections"],
    )

    # Stage 2: Merge CSVs with bounding boxes
    print("\n--- Stage 2: Merging CSVs with Bounding Boxes ---")
    merge_csvs_with_bbox(
        config["csv"]["merged_detections"],
        config["csv"]["metadata"],
        config["csv"]["merged_bbox"],
    )

    # Stage 3: Update JSON annotations with timestamps
    print("\n--- Stage 3: Updating JSON Annotations with Timestamps ---")
    update_json_with_timestamp(
        config["json"]["left"]["input"],
        config["csv"]["merged_bbox"],
        config["json"]["left"]["output"],
    )
    update_json_with_timestamp(
        config["json"]["right"]["input"],
        config["csv"]["merged_bbox"],
        config["json"]["right"]["output"],
    )

    # Load updated JSON data
    with open(config["json"]["left"]["output"], "r") as f:
        data_left = json.load(f)
    with open(config["json"]["right"]["output"], "r") as f:
        data_right = json.load(f)

    print("\nInitial Left Viewpoint Cluster Summary:")
    print_cluster_summary(data_left)
    print_viewpoint_cluster_mapping(data_left, "left")

    print("\nInitial Right Viewpoint Cluster Summary:")
    print_cluster_summary(data_right)
    print_viewpoint_cluster_mapping(data_right, "right")

    # Stage 4: Split Cluster Verification
    print("\n--- Stage 4: Split Cluster Verification ---")
    grouped_left = group_annotations_by_LCA_all(data_left)
    grouped_right = group_annotations_by_LCA_all(data_right)

    iteration = 0
    while True:
        iteration += 1
        print(f"\n--- Split Cluster Verification Iteration: {iteration} ---")
        
        pair_ids_needing_decisions = consistency_check_with_database(
            grouped_left, grouped_right, data_left, data_right, image_dir, db_path
        )
        
        if not pair_ids_needing_decisions:
            print("No split clusters found. Proceeding to the next stage.")
            break
        
        wait_for_database_decisions(db_path, pair_ids_needing_decisions)
        process_split_decisions(grouped_left, grouped_right, data_left, data_right, db_path)
        
        grouped_left = group_annotations_by_LCA_all(data_left)
        grouped_right = group_annotations_by_LCA_all(data_right)

    left_split_file = save_json_with_stage(data_left, config["json"]["left"]["output"], "split")
    right_split_file = save_json_with_stage(data_right, config["json"]["right"]["output"], "split")

    # Stage 6: Time-Overlap Verification
    print("\n--- Stage 6: Time-Overlap Verification ---")
    left_pair_ids = time_overlap_verification_db(grouped_left, data_left, image_dir, db_path, "left")
    right_pair_ids = time_overlap_verification_db(grouped_right, data_right, image_dir, db_path, "right")

    all_time_pair_ids = left_pair_ids + right_pair_ids
    if all_time_pair_ids:
        wait_for_database_decisions(db_path, all_time_pair_ids)
        process_time_decisions(grouped_left, data_left, db_path, "left")
        process_time_decisions(grouped_right, data_right, db_path, "right")

    left_time_file = save_json_with_stage(data_left, left_split_file, "time")
    right_time_file = save_json_with_stage(data_right, right_split_file, "time")

    # Stage 5 & 7: Cluster Equivalence and ID Assignment
    print("\n--- Stage 5: Cluster Equivalence ---")
    grouped_left_wv = group_annotations_by_LCA_with_viewpoint(data_left, "left")
    grouped_right_wv = group_annotations_by_LCA_with_viewpoint(data_right, "right")
    check_cluster_equivalence(grouped_left_wv, grouped_right_wv)

    print("\n--- Stage 7: ID Assignment ---")
    assign_ids_after_equivalence_check(data_left, data_right)

    left_final_file = save_json_with_stage(data_left, left_time_file, "final_with_ids")
    right_final_file = save_json_with_stage(data_right, right_time_file, "final_with_ids")

    print("\nFinal Cluster Summary - Left VP:")
    print_cluster_summary(data_left)
    print("\nFinal Cluster Summary - Right VP:")
    print_cluster_summary(data_right)

    print("\nAll stages completed.")
    print(f"\nDatabase contains verification tasks at: {os.path.abspath(db_path)}")
    print("Use the UI handler to process verification tasks.")


if __name__ == "__main__":
    main()