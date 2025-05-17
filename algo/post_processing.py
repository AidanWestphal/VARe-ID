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
from collections import defaultdict
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from PIL import Image

# If running in a Jupyter Notebook, enable inline plotting.
try:
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass

# Try to import ipywidgets and IPython.display for interactive decisions.
try:
    import ipywidgets as widgets
    from IPython.display import display
except ImportError:
    widgets = None
    print(
        "ipywidgets not available; falling back to console input for interactive decisions."
    )


# -------------------------
# Helper function for interactive waiting using ipywidgets.
# -------------------------
def wait_for_dropdown_value(dropdown):
    """Block until the dropdown value is changed from 'Select'."""
    while dropdown.value == "Select":
        time.sleep(0.1)
    return dropdown.value


# -------------------------
# Helper: Get user decision (using either widgets or input)
# -------------------------
def get_user_decision(prompt="Merge clusters? (Yes/No): ", interactive=True):
    if interactive and widgets is not None:
        dropdown = widgets.Dropdown(
            options=["Select", "Yes", "No"],
            value="Select",
            description=prompt,
            style={"description_width": "initial"},
        )
        display(dropdown)
        decision = wait_for_dropdown_value(dropdown)
        return decision
    else:
        decision = input(prompt).strip().lower()
        return "Yes" if decision.startswith("y") else "No"


# -------------------------
# Helper: Save JSON file with stage suffix.
# -------------------------
def save_json_with_stage(data, original_filename, stage_suffix):
    base, ext = os.path.splitext(original_filename)
    new_filename = base + "_" + stage_suffix + ext
    with open(new_filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved file: {new_filename}")
    return new_filename


# -------------------------
# Helper: Print cluster summary for a JSON data file.
# -------------------------
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
            print(f"  Tracking ID: {ann.get('tracking_id')}, CA_score: {ca:.8f}")
        print()


# -------------------------
# Helper: Print viewpoint cluster-to-tracking ID mapping.
# -------------------------
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
# Stage 4: Split Cluster Verification (Interactive)
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
    for img in images:
        if img["uuid"] == image_uuid:
            return os.path.join(image_dir, img["file_name"])
    return None


def display_image(data, annotation, image_dir):
    image_uuid = annotation["image_uuid"]
    file_path = get_image_filename(data["images"], image_uuid, image_dir)
    if file_path and os.path.exists(file_path):
        try:
            image = Image.open(file_path)
            bbox = annotation["bbox"]
            fig = plt.figure()
            cropped = image.crop(
                (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
            )
            plt.imshow(cropped)
            plt.title(
                f'Cluster: {annotation["LCA_clustering_id"]}\nCA Score: {annotation["CA_score"]}'
            )
            plt.axis("off")
            display(fig)
            plt.close(fig)
        except Exception as e:
            print(f"Error displaying image: {e}")
    else:
        print(f"Image file not found: {file_path}")


# -------------------------
# Split / time-overlap update functions
# -------------------------
def update_split_cluster(data, grouped_annotations, cluster1, cluster2, merge=True):
    """
    If merging, move cluster1’s annotations into cluster2, then delete cluster1.
    Otherwise, update the tracking IDs if there's overlap.
    """
    if merge:
        # Merge cluster1 into cluster2
        grouped_annotations[cluster2].extend(grouped_annotations[cluster1])
        for ann in grouped_annotations[cluster1]:
            ann["LCA_clustering_id"] = cluster2
        del grouped_annotations[cluster1]
        print(
            f"[Split Stage] Clusters {cluster1} and {cluster2} merged into {cluster2}."
        )
    else:
        ids1 = set(ann["tracking_id"] for ann in grouped_annotations[cluster1])
        ids2 = set(ann["tracking_id"] for ann in grouped_annotations[cluster2])
        common_ids = ids1.intersection(ids2)
        if common_ids:
            for ann in grouped_annotations[cluster2]:
                if ann["tracking_id"] in common_ids:
                    ann["tracking_id"] = str(ann["tracking_id"]) + "_new"
            print(
                f"[Split Stage] Updated common tracking IDs {common_ids} in cluster {cluster2} to include '_new'."
            )
        else:
            print("[Split Stage] No common tracking IDs found; no update performed.")


def update_time_overlap(data, grouped_annotations, cluster1, cluster2, merge=True):
    """If merging, move cluster2's annotations to cluster1. Otherwise, do nothing."""
    if merge:
        for ann in grouped_annotations[cluster2]:
            ann["LCA_clustering_id"] = cluster1
        grouped_annotations[cluster1].extend(grouped_annotations[cluster2])
        del grouped_annotations[cluster2]
        print(
            f"[Time Stage] Clusters {cluster1} and {cluster2} merged into {cluster1}."
        )
    else:
        print(
            f"[Time Stage] No merging performed for clusters {cluster1} and {cluster2}."
        )


# -------------------------
# Pairwise verification (returns True if merge happened, else False)
# -------------------------
def pairwise_verification_with_update(
    grouped_annotations,
    cluster1,
    cluster2,
    data,
    image_dir,
    interactive=True,
    stage="split",
):
    if cluster1 not in grouped_annotations or cluster2 not in grouped_annotations:
        print(
            f"One of the clusters {cluster1} or {cluster2} is not present. Skipping verification."
        )
        return False

    if not grouped_annotations[cluster1] or not grouped_annotations[cluster2]:
        print(
            f"One of the clusters {cluster1} or {cluster2} is empty. Skipping verification."
        )
        return False

    best_ann1 = max(grouped_annotations[cluster1], key=lambda ann: ann["CA_score"])
    best_ann2 = max(grouped_annotations[cluster2], key=lambda ann: ann["CA_score"])

    print(f"\nVerifying clusters: {cluster1} and {cluster2}")
    print(
        f"  Annotation 1 (UUID: {best_ann1['uuid']}): CA_score = {best_ann1['CA_score']}"
    )
    print(
        f"  Annotation 2 (UUID: {best_ann2['uuid']}): CA_score = {best_ann2['CA_score']}"
    )
    display_image(data, best_ann1, image_dir)
    display_image(data, best_ann2, image_dir)

    decision = get_user_decision(
        prompt="Merge clusters? (Yes/No): ", interactive=interactive
    )

    if stage == "split":
        if decision == "Yes":
            update_split_cluster(
                data, grouped_annotations, cluster1, cluster2, merge=True
            )
            return True
        else:
            update_split_cluster(
                data, grouped_annotations, cluster1, cluster2, merge=False
            )
            return False

    elif stage == "time":
        if decision == "Yes":
            update_time_overlap(
                data, grouped_annotations, cluster1, cluster2, merge=True
            )
            return True
        else:
            update_time_overlap(
                data, grouped_annotations, cluster1, cluster2, merge=False
            )
            return False


# -------------------------
# Dynamic "split" consistency check
# -------------------------
def consistency_check_with_updates(
    grouped_left, grouped_right, data_left, data_right, image_dir, interactive=True
):
    """
    Dynamically detects "split clusters" by scanning each viewpoint until stable.
    """
    split_found = False

    # 1) Left viewpoint
    while True:
        left_tracking = defaultdict(set)
        for lca, anns in grouped_left.items():
            for ann in anns:
                left_tracking[ann["tracking_id"]].add(lca)

        any_merge = False

        for tid, clusters in left_tracking.items():
            if len(clusters) > 1:
                split_found = True
                print(f"\n[Left VP] Tracking ID {tid} has split clusters: {clusters}")
                cl_list = list(clusters)

                i = 0
                while i < len(cl_list) - 1:
                    c1 = cl_list[i]
                    c2 = cl_list[i + 1]
                    print(f"Evaluating clusters {c1} and {c2} for TID {tid}")

                    did_merge = pairwise_verification_with_update(
                        grouped_left,
                        c1,
                        c2,
                        data_left,
                        image_dir,
                        interactive,
                        stage="split",
                    )
                    if did_merge:
                        any_merge = True
                        break
                    else:
                        # If "No," we rename TIDs if needed, then keep going
                        cl_list = [c for c in cl_list if c in grouped_left]
                        i += 1

            if any_merge:
                break

        if not any_merge:
            break

    # 2) Right viewpoint
    while True:
        right_tracking = defaultdict(set)
        for lca, anns in grouped_right.items():
            for ann in anns:
                right_tracking[ann["tracking_id"]].add(lca)

        any_merge = False

        for tid, clusters in right_tracking.items():
            if len(clusters) > 1:
                split_found = True
                print(f"\n[Right VP] Tracking ID {tid} has split clusters: {clusters}")
                cl_list = list(clusters)

                i = 0
                while i < len(cl_list) - 1:
                    c1 = cl_list[i]
                    c2 = cl_list[i + 1]
                    print(f"Evaluating clusters {c1} and {c2} for TID {tid}")

                    did_merge = pairwise_verification_with_update(
                        grouped_right,
                        c1,
                        c2,
                        data_right,
                        image_dir,
                        interactive,
                        stage="split",
                    )
                    if did_merge:
                        any_merge = True
                        break
                    else:
                        cl_list = [c for c in cl_list if c in grouped_right]
                        i += 1

            if any_merge:
                break

        if not any_merge:
            break

    # 3) Final summary
    left_tracking = defaultdict(set)
    for lca, anns in grouped_left.items():
        for ann in anns:
            left_tracking[ann["tracking_id"]].add(lca)

    right_tracking = defaultdict(set)
    for lca, anns in grouped_right.items():
        for ann in anns:
            right_tracking[ann["tracking_id"]].add(lca)

    print("\n[Split Stage] Final consistency check (tracking ID -> clusters):")
    all_tids = set(left_tracking.keys()).union(right_tracking.keys())
    for tid in all_tids:
        lcs = left_tracking.get(tid, set())
        rcs = right_tracking.get(tid, set())
        print(f"Tracking ID {tid} -> Left clusters: {lcs} | Right clusters: {rcs}")
        if len(lcs) == len(rcs):
            print("  Status: Consistent")
        else:
            print("  Status: Inconsistent")

    return split_found


# =========================
# Stage 6: Time-Overlap Verification
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


def find_clusters_with_no_time_overlap(
    grouped_annotations, threshold=timedelta(seconds=1)
):
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


def time_overlap_verification(grouped_annotations, data, image_dir, interactive=True):
    print("\n[Stage 6] Checking for cluster pairs with NO time overlap...")
    no_overlap_pairs = find_clusters_with_no_time_overlap(grouped_annotations)
    if not no_overlap_pairs:
        print("All clusters have some time overlap.")
        return

    print(f"Found {len(no_overlap_pairs)} cluster pair(s) with no time overlap.")
    for c1, c2 in no_overlap_pairs:
        print(f"\nCluster pair with NO overlap: {c1} & {c2}")
        pairwise_verification_with_update(
            grouped_annotations, c1, c2, data, image_dir, interactive, stage="time"
        )


# =========================
# Stage 5: Cluster Equivalence (No ID assignment)
# =========================
def group_annotations_by_LCA_with_viewpoint(data, viewpoint):
    grouped = defaultdict(list)
    for ann in data["annotations"]:
        lca = ann.get("LCA_clustering_id")
        if lca is not None:
            grouped[f"{lca}_{viewpoint}"].append(ann)
    return grouped


def check_cluster_equivalence(grouped_left, grouped_right):
    """
    Here we just check if clusters appear to match across viewpoints.
    (No individual ID assignment is done here.)
    """
    print("\nStage 5: Checking Cluster Equivalence (No 'individual_id' usage)")
    print("================================================================\n")

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
        names = {rc.rsplit("_", 1)[0] for rc in rcs}
        if len(names) > 1:
            print(
                f"Cluster {left_cl.rsplit('_',1)[0]} in Left VP spans multiple clusters in Right VP: {names}"
            )
        else:
            print(
                f"Cluster {left_cl.rsplit('_',1)[0]} in Left VP is equivalent to Right VP cluster(s): {names}"
            )

    for right_cl, anns in grouped_right.items():
        right_tids = {ann["tracking_id"] for ann in anns}
        lcs = {tracking_left[tid] for tid in right_tids if tid in tracking_left}
        equiv["right_to_left"][right_cl] = lcs
        names = {lc.rsplit("_", 1)[0] for lc in lcs}
        if len(names) > 1:
            print(
                f"Cluster {right_cl.rsplit('_',1)[0]} in Right VP spans multiple clusters in Left VP: {names}"
            )
        else:
            print(
                f"Cluster {right_cl.rsplit('_',1)[0]} in Right VP is equivalent to Left VP cluster(s): {names}"
            )

    print(
        "\nStage 5 Completed: Cluster Equivalence Check Done (no ID assignment performed).\n"
    )
    return equiv


# -------------------------
# OPTIONAL: Stage 7 – ID Assignment After Equivalence
# -------------------------
def assign_ids_after_equivalence_check(data_left, data_right):
    """
    Assign 'individual_id' values after finding equivalences between
    left and right viewpoint clusters. If a left cluster and a right cluster
    share any tracking IDs, we group them under one 'individual_id'.
    """
    # Re-use the same viewpoint grouping + equivalence check:
    grouped_left = group_annotations_by_LCA_with_viewpoint(data_left, "left")
    grouped_right = group_annotations_by_LCA_with_viewpoint(data_right, "right")
    equiv = check_cluster_equivalence(grouped_left, grouped_right)

    # Build sets of "equivalent" cluster groups
    individual_id = 1
    processed = set()
    eq_groups = []

    for left_cl, right_cls in equiv["left_to_right"].items():
        if right_cls:
            group = set([left_cl]) | right_cls
            eq_groups.append(group)

    # Merge any overlapping sets in eq_groups
    merged_groups = []
    while eq_groups:
        first = eq_groups.pop(0)
        merged = False
        for i, grp in enumerate(merged_groups):
            # If there's an overlap, unify them
            if first & grp:
                merged_groups[i] = grp | first
                merged = True
                break
        if not merged:
            merged_groups.append(first)

    # Assign a single individual_id to each merged group
    for group in merged_groups:
        clusters = group - processed
        if clusters:
            # If the group has clusters from multiple viewpoints, we treat them as same ID
            views = {cl.split("_")[-1] for cl in clusters}
            if len(views) > 1:
                names = {cl.rsplit("_", 1)[0] for cl in clusters}
                print(
                    f"Equivalent Clusters {names}. Assigning Individual ID: {individual_id}"
                )
                for cl in clusters:
                    if cl.endswith("_left"):
                        for ann in grouped_left[cl]:
                            ann["individual_id"] = individual_id
                    else:
                        for ann in grouped_right[cl]:
                            ann["individual_id"] = individual_id
                    processed.add(cl)
                individual_id += 1
            else:
                # cluster is in only one viewpoint
                for cl in clusters:
                    name = cl.rsplit("_", 1)[0]
                    print(
                        f"Unmatched: {cl} in {'Left' if cl.endswith('_left') else 'Right'} VP. "
                        f"Assigning Temporary ID: temp_{name}"
                    )
                    if cl.endswith("_left"):
                        for ann in grouped_left[cl]:
                            ann["individual_id"] = f"temp_{name}"
                    else:
                        for ann in grouped_right[cl]:
                            ann["individual_id"] = f"temp_{name}"
                    processed.add(cl)

    # Also handle any leftover clusters not yet processed
    for cl, anns in grouped_left.items():
        if cl not in processed:
            name = cl.rsplit("_", 1)[0]
            print(
                f"Unmatched: Left Cluster {name}. Assigning Temporary ID: temp_{name}"
            )
            for ann in anns:
                ann["individual_id"] = f"temp_{name}"
            processed.add(cl)

    for cl, anns in grouped_right.items():
        if cl not in processed:
            name = cl.rsplit("_", 1)[0]
            print(
                f"Unmatched: Right Cluster {name}. Assigning Temporary ID: temp_{name}"
            )
            for ann in anns:
                ann["individual_id"] = f"temp_{name}"
            processed.add(cl)

    print("[Stage 7] ID Assignment Completed.")


# -------------------------
# Utility JSON I/O Functions
# -------------------------
def load_json_data(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def save_json_data(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


# =========================
# Main Workflow: Chain All Stages Together
# =========================
def main():
    with open("post_processing.yaml", "r") as f:
        config = yaml.safe_load(f)
    interactive = config.get("interactive", True)

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

    # Load updated JSON data and print initial cluster information.
    data_left = load_json_data(config["json"]["left"]["output"])
    data_right = load_json_data(config["json"]["right"]["output"])

    print("\nInitial Left Viewpoint Cluster Summary:")
    print_cluster_summary(data_left)
    print_viewpoint_cluster_mapping(data_left, "left")

    print("\nInitial Right Viewpoint Cluster Summary:")
    print_cluster_summary(data_right)
    print_viewpoint_cluster_mapping(data_right, "right")

    # Stage 4: Split Cluster Verification (Iterative)
    print("\n--- Stage 4: Split Cluster Verification (Iterative) ---")
    grouped_left = group_annotations_by_LCA_all(data_left)
    grouped_right = group_annotations_by_LCA_all(data_right)

    iteration = 0
    while True:
        iteration += 1
        print(f"\n--- Split Cluster Verification Iteration: {iteration} ---")
        split_found = consistency_check_with_updates(
            grouped_left,
            grouped_right,
            data_left,
            data_right,
            config["image"]["directory"],
            interactive,
        )
        if not split_found:
            print("No split clusters found. Proceeding to the next stage.")
            break
        # Refresh the groupings after updates
        grouped_left = group_annotations_by_LCA_all(data_left)
        grouped_right = group_annotations_by_LCA_all(data_right)

    left_split_file = save_json_with_stage(
        data_left, config["json"]["left"]["output"], "split"
    )
    right_split_file = save_json_with_stage(
        data_right, config["json"]["right"]["output"], "split"
    )

    print("\nAfter Split Cluster Stage - Left VP:")
    print_cluster_summary(data_left)
    print("\nAfter Split Cluster Stage - Right VP:")
    print_cluster_summary(data_right)

    # Stage 6: Time-Overlap Verification
    print("\n--- Stage 6: Time-Overlap Verification ---")
    print("\nLeft viewpoint:")
    time_overlap_verification(
        grouped_left, data_left, config["image"]["directory"], interactive
    )

    print("\nRight viewpoint:")
    time_overlap_verification(
        grouped_right, data_right, config["image"]["directory"], interactive
    )

    left_time_file = save_json_with_stage(data_left, left_split_file, "time")
    right_time_file = save_json_with_stage(data_right, right_split_file, "time")

    print("\nAfter Time-Overlap Stage - Left VP:")
    print_cluster_summary(data_left)
    print("\nAfter Time-Overlap Stage - Right VP:")
    print_cluster_summary(data_right)

    # Stage 5: Cluster Equivalence (No 'individual_id' Assignment)
    print("\n--- Stage 5: Cluster Equivalence (No 'individual_id' Assignment) ---")
    print_viewpoint_cluster_mapping(data_left, "left")
    print_viewpoint_cluster_mapping(data_right, "right")

    grouped_left_wv = group_annotations_by_LCA_with_viewpoint(data_left, "left")
    grouped_right_wv = group_annotations_by_LCA_with_viewpoint(data_right, "right")
    check_cluster_equivalence(grouped_left_wv, grouped_right_wv)

    # -------------------------
    # Stage 7: ID Assignment
    # -------------------------
    print("\n--- Stage 7: Assigning IDs after equivalence check ---")
    assign_ids_after_equivalence_check(data_left, data_right)

    left_final_file = save_json_with_stage(data_left, left_time_file, "final_with_ids")
    right_final_file = save_json_with_stage(
        data_right, right_time_file, "final_with_ids"
    )

    print("\nFinal Cluster Summary - Left VP:")
    print_cluster_summary(data_left)
    print("\nFinal Cluster Summary - Right VP:")
    print_cluster_summary(data_right)

    print("\nAll stages completed.")


if __name__ == "__main__":
    main()
