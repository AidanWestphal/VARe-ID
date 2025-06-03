# #!/usr/bin/env python3
# import os
# import sys
# import time
# import yaml
# import pandas as pd
# import csv
# import re
# import ast
# import json
# from collections import defaultdict
# from datetime import datetime, timedelta
# import matplotlib.pyplot as plt
# from PIL import Image

# # If running in a Jupyter Notebook, enable inline plotting.
# try:
#     get_ipython().run_line_magic("matplotlib", "inline")
# except Exception:
#     pass

# # Try to import ipywidgets and IPython.display for interactive decisions.
# try:
#     import ipywidgets as widgets
#     from IPython.display import display
# except ImportError:
#     widgets = None
#     print("ipywidgets not available; falling back to console input for interactive decisions.")

# # -------------------------
# # Helper function for interactive waiting using ipywidgets.
# # -------------------------
# def wait_for_dropdown_value(dropdown):
#     """Block until the dropdown value is changed from 'Select'."""
#     while dropdown.value == 'Select':
#         time.sleep(0.1)
#     return dropdown.value

# # -------------------------
# # Helper: Get user decision (using either widgets or input)
# # -------------------------
# def get_user_decision(prompt="Merge clusters? (Yes/No): ", interactive=True):
#     if interactive and widgets is not None:
#         dropdown = widgets.Dropdown(
#             options=['Select', 'Yes', 'No'],
#             value='Select',
#             description=prompt,
#             style={'description_width': 'initial'}
#         )
#         display(dropdown)
#         decision = wait_for_dropdown_value(dropdown)
#         return decision
#     else:
#         decision = input(prompt).strip().lower()
#         return "Yes" if decision.startswith("y") else "No"

# # -------------------------
# # Helper: Save JSON file with stage suffix.
# # -------------------------
# def save_json_with_stage(data, original_filename, stage_suffix):
#     base, ext = os.path.splitext(original_filename)
#     new_filename = base + "_" + stage_suffix + ext
#     with open(new_filename, "w") as f:
#         json.dump(data, f, indent=4)
#     print(f"Saved file: {new_filename}")
#     return new_filename

# # -------------------------
# # Helper: Print cluster summary for a JSON data file.
# # -------------------------
# def print_cluster_summary(data):
#     annotations = data.get("annotations", [])
#     total = len(annotations)
#     grouped = defaultdict(list)
#     for ann in annotations:
#         key = ann.get("LCA_clustering_id")
#         if key is not None:
#             grouped[key].append(ann)
#     unique = len(grouped)
#     print(f"\nTotal count of annotations: {total}")
#     print(f"Count of unique clustering IDs: {unique}\n")
#     for cluster in sorted(grouped.keys()):
#         print(f"LCA_clustering_id: {cluster}")
#         for ann in grouped[cluster]:
#             ca = ann.get("CA_score", 0)
#             print(
#                 f"Tracking ID: {ann.get('tracking_id')}, Individual ID: {ann.get('individual_id')}, "
#                 f"CA_score: {ca:.8f}"
#             )
#         print()

# # -------------------------
# # Helper: Print viewpoint cluster-to-tracking ID mapping.
# # -------------------------
# def print_viewpoint_cluster_mapping(data, viewpoint):
#     annotations = data.get("annotations", [])
#     grouped = defaultdict(set)
#     for ann in annotations:
#         key = ann.get("LCA_clustering_id")
#         if key is not None:
#             grouped[key].add(ann.get("tracking_id"))
#     print(f"\n{viewpoint.capitalize()} Viewpoint Cluster to Tracking ID Mapping:")
#     for cluster in sorted(grouped.keys()):
#         print(f"  Cluster {cluster}_{viewpoint}: Tracking IDs: {grouped[cluster]}")

# # =========================
# # Stage 1: Merge Detection CSV Files
# # =========================
# def merge_csv_files(file1_path, file2_path, output_path):
#     df1 = pd.read_csv(file1_path)
#     df2 = pd.read_csv(file2_path)
#     merged_df = pd.concat([df1, df2], ignore_index=True)
#     merged_df = merged_df.sort_values('timestamp')
#     merged_df.to_csv(output_path, index=False)
#     print(f"[Stage 1] Merged detection CSV files into: {output_path}")
#     print(f"[Stage 1] Total rows: {len(merged_df)}")

# # =========================
# # Stage 2: Merge CSVs with Bounding Boxes
# # =========================
# def parse_bbox_string_xyxy(bbox_str):
#     s = bbox_str.strip()
#     s = re.sub(r'[\[\]\(\)]', '', s)
#     parts = s.split(',')
#     try:
#         return tuple(int(p.strip()) for p in parts)
#     except Exception as e:
#         print(f"Error parsing bbox (xyxy): {bbox_str} -> {e}")
#         return None

# def parse_bbox_string_xywh(bbox_str):
#     s = bbox_str.strip()
#     s = re.sub(r'[\[\]\(\)]', '', s)
#     parts = s.split(',')
#     try:
#         coords = [int(p.strip()) for p in parts]
#         if len(coords) != 4:
#             return None
#         x1, y1, w, h = coords
#         return (x1, y1, x1 + w, y1 + h)
#     except Exception as e:
#         print(f"Error parsing bbox (xywh): {bbox_str} -> {e}")
#         return None

# def merge_csvs_with_bbox(timestamps_csv, metadata_csv, output_csv):
#     dict_ts = {}
#     with open(timestamps_csv, 'r', newline='', encoding='utf-8') as f1:
#         reader1 = csv.DictReader(f1)
#         ts_fieldnames = reader1.fieldnames or []
#         for row in reader1:
#             ts_filename = row.get('frame_name')
#             if not ts_filename:
#                 continue
#             ts_bbox_str = row.get('bounding_box')
#             if not ts_bbox_str:
#                 continue
#             ts_bbox = parse_bbox_string_xyxy(ts_bbox_str)
#             if ts_bbox is None:
#                 continue
#             key = (ts_filename, ts_bbox)
#             dict_ts[key] = row

#     merged_rows = []
#     combined_fieldnames = set(ts_fieldnames)

#     with open(metadata_csv, 'r', newline='', encoding='utf-8') as f2:
#         reader2 = csv.DictReader(f2)
#         metadata_fieldnames = reader2.fieldnames or []
#         for col in metadata_fieldnames:
#             combined_fieldnames.add(col)

#         for row in reader2:
#             md_filename = row.get('file_name')
#             if not md_filename:
#                 continue
#             md_bbox_str = row.get('bbox')
#             if not md_bbox_str:
#                 continue
#             md_bbox = parse_bbox_string_xywh(md_bbox_str)
#             if md_bbox is None:
#                 continue
#             key = (md_filename, md_bbox)
#             if key in dict_ts:
#                 ts_row = dict_ts[key]
#                 merged_dict = {}
#                 merged_dict.update(ts_row)
#                 merged_dict.update(row)
#                 merged_rows.append(merged_dict)

#     combined_fieldnames = list(combined_fieldnames)
#     with open(output_csv, 'w', newline='', encoding='utf-8') as out:
#         writer = csv.DictWriter(out, fieldnames=combined_fieldnames)
#         writer.writeheader()
#         for row in merged_rows:
#             writer.writerow(row)

#     print(f"[Stage 2] Merged CSV with bounding boxes written to: {output_csv}")
#     print(f"[Stage 2] Total merged rows: {len(merged_rows)}")

# # =========================
# # Stage 3: Update JSON Annotations with Timestamps
# # =========================
# def parse_bbox(bbox_str):
#     try:
#         return list(ast.literal_eval(bbox_str))
#     except Exception as e:
#         print(f"Error parsing bbox from CSV: {bbox_str} -> {e}")
#         return None

# def make_comparable_dict_from_csv(row, file_to_uuid):
#     # Attempt to parse fields in CSV
#     try:
#         tracking_id = int(row["tracking_id"].strip()) if row["tracking_id"].strip() else None
#     except:
#         tracking_id = None
#     try:
#         confidence = float(row["confidence"].strip()) if row["confidence"].strip() else None
#     except:
#         confidence = None
#     try:
#         detection_class = int(row["detection_class"].strip()) if row["detection_class"].strip() else None
#     except:
#         detection_class = None
#     try:
#         CA_score = float(row["CA_score"].strip()) if row["CA_score"].strip() else None
#     except:
#         CA_score = None
#     try:
#         individual_id = int(row["individual_id"].strip()) if row["individual_id"].strip() else None
#     except:
#         individual_id = None

#     species = row["species"].strip() if "species" in row else None
#     bbox = parse_bbox(row["bbox"])
#     image_uuid = file_to_uuid.get(row["file_name"].strip(), None)
#     timestamp = row["timestamp"].strip()

#     comparable_fields = {
#         "image_uuid": image_uuid,
#         "tracking_id": tracking_id,
#         "confidence": confidence,
#         "detection_class": detection_class,
#         "species": species,
#         "bbox": bbox,
#         "individual_id": individual_id,
#         "CA_score": CA_score,
#     }
#     return comparable_fields, timestamp

# def make_comparable_dict_from_json(ann):
#     bbox_list = list(ann["bbox"]) if ann.get("bbox") else None
#     return {
#         "image_uuid": ann["image_uuid"],
#         "tracking_id": ann["tracking_id"],
#         "confidence": ann["confidence"],
#         "detection_class": ann["detection_class"],
#         "species": ann["species"],
#         "bbox": bbox_list,
#         "individual_id": ann["individual_id"],
#         "CA_score": ann["CA_score"],
#     }

# def update_json_with_timestamp(json_input, csv_input, json_output):
#     with open(json_input, "r") as f:
#         data = json.load(f)

#     file_to_uuid = {img["file_name"]: img["uuid"] for img in data["images"]}
#     csv_common_list = []

#     with open(csv_input, "r", newline="") as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             comp_fields, ts = make_comparable_dict_from_csv(row, file_to_uuid)
#             if comp_fields["image_uuid"] is not None:
#                 csv_common_list.append((comp_fields, ts))

#     updated_annotations = []
#     for (csv_fields, csv_timestamp) in csv_common_list:
#         for ann in data["annotations"]:
#             ann_fields = make_comparable_dict_from_json(ann)
#             if ann_fields == csv_fields:
#                 ann["timestamp"] = csv_timestamp
#                 updated_annotations.append({
#                     "annotation_uuid": ann["uuid"],
#                     "image_uuid": ann["image_uuid"],
#                     "tracking_id": ann["tracking_id"],
#                     "timestamp_added": csv_timestamp
#                 })
#                 break

#     with open(json_output, "w") as f:
#         json.dump(data, f, indent=4)

#     print(f"[Stage 3] Updated JSON written to: {json_output}")
#     print(f"[Stage 3] Number of annotations updated: {len(updated_annotations)}")

# # =========================
# # Stage 4: Split Cluster Verification (Interactive)
# # =========================

# def group_annotations_by_LCA_all(data):
#     annotations = data['annotations']
#     grouped = defaultdict(list)
#     for ann in annotations:
#         lca = ann.get('LCA_clustering_id')
#         if lca is not None:
#             grouped[lca].append(ann)
#     return grouped

# def get_image_filename(images, image_uuid, image_dir):
#     for img in images:
#         if img['uuid'] == image_uuid:
#             return os.path.join(image_dir, img['file_name'])
#     return None

# def display_image(data, annotation, image_dir):
#     image_uuid = annotation['image_uuid']
#     file_path = get_image_filename(data['images'], image_uuid, image_dir)
#     if file_path and os.path.exists(file_path):
#         try:
#             image = Image.open(file_path)
#             bbox = annotation['bbox']
#             fig = plt.figure()
#             cropped = image.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
#             plt.imshow(cropped)
#             plt.title(f'Cluster: {annotation["LCA_clustering_id"]}\nCA Score: {annotation["CA_score"]}')
#             plt.axis('off')
#             display(fig)
#             plt.close(fig)
#         except Exception as e:
#             print(f"Error displaying image: {e}")
#     else:
#         print(f"Image file not found: {file_path}")

# # -------------------------
# # Updates to the "split" vs "time" cluster merges
# # -------------------------
# # -------------------------
# # Split / time-overlap update functions
# # -------------------------
# def update_split_cluster(data, grouped_annotations, cluster1, cluster2, merge=True):
#     """
#     If merging, move cluster1’s annotations into cluster2, then delete cluster1.
#     Otherwise, update the tracking IDs if there's overlap.
#     """
#     if merge:
#         # Merge cluster1 into cluster2
#         grouped_annotations[cluster2].extend(grouped_annotations[cluster1])
#         for ann in grouped_annotations[cluster1]:
#             ann['LCA_clustering_id'] = cluster2
#         del grouped_annotations[cluster1]
#         print(f"[Split Stage] Clusters {cluster1} and {cluster2} merged into {cluster2}.")
#     else:
#         ids1 = set(ann['tracking_id'] for ann in grouped_annotations[cluster1])
#         ids2 = set(ann['tracking_id'] for ann in grouped_annotations[cluster2])
#         common_ids = ids1.intersection(ids2)
#         if common_ids:
#             for ann in grouped_annotations[cluster2]:
#                 if ann['tracking_id'] in common_ids:
#                     ann['tracking_id'] = str(ann['tracking_id']) + "_new"
#             print(f"[Split Stage] Updated common tracking IDs {common_ids} in cluster {cluster2} to include '_new'.")
#         else:
#             print("[Split Stage] No common tracking IDs found; no update performed.")



# def update_time_overlap(data, grouped_annotations, cluster1, cluster2, merge=True):
#     """
#     If merge=True, fully combine cluster2 into cluster1 and remove cluster2.
#     Otherwise, do nothing special.
#     """
#     if merge:
#         for ann in grouped_annotations[cluster2]:
#             ann['LCA_clustering_id'] = cluster1
#         grouped_annotations[cluster1].extend(grouped_annotations[cluster2])
#         del grouped_annotations[cluster2]
#         print(f"[Time Stage] Clusters {cluster1} and {cluster2} merged into {cluster1}.")
#     else:
#         print(f"[Time Stage] No merging performed for clusters {cluster1} and {cluster2}.")

# # -------------------------
# # The pairwise verification, now returning a bool to indicate merge or not
# # -------------------------
# def pairwise_verification_with_update(
#     grouped_annotations, cluster1, cluster2, data, image_dir,
#     interactive=True, stage="split"
# ):
#     # Skip if cluster missing or empty
#     if cluster1 not in grouped_annotations or cluster2 not in grouped_annotations:
#         print(f"One of the clusters {cluster1} or {cluster2} is not present. Skipping verification.")
#         return False

#     if not grouped_annotations[cluster1] or not grouped_annotations[cluster2]:
#         print(f"One of the clusters {cluster1} or {cluster2} is empty. Skipping verification.")
#         return False

#     # Choose top CA_score annotation for display
#     best_ann1 = max(grouped_annotations[cluster1], key=lambda ann: ann['CA_score'])
#     best_ann2 = max(grouped_annotations[cluster2], key=lambda ann: ann['CA_score'])

#     print(f"\nVerifying clusters: {cluster1} and {cluster2}")
#     print(f"  Annotation 1 (UUID: {best_ann1['uuid']}): CA_score = {best_ann1['CA_score']}")
#     print(f"  Annotation 2 (UUID: {best_ann2['uuid']}): CA_score = {best_ann2['CA_score']}")
#     display_image(data, best_ann1, image_dir)
#     display_image(data, best_ann2, image_dir)

#     decision = get_user_decision(prompt="Merge clusters? (Yes/No): ", interactive=interactive)

#     if stage == "split":
#         if decision == 'Yes':
#             update_split_cluster(data, grouped_annotations, cluster1, cluster2, merge=True)
#             return True  # A MERGE happened
#         else:
#             update_split_cluster(data, grouped_annotations, cluster1, cluster2, merge=False)
#             return False # NO MERGE
#     elif stage == "time":
#         if decision == 'Yes':
#             update_time_overlap(data, grouped_annotations, cluster1, cluster2, merge=True)
#             return True
#         else:
#             update_time_overlap(data, grouped_annotations, cluster1, cluster2, merge=False)
#             return False

# # -------------------------
# # Dynamic version of consistency_check_with_updates
# # -------------------------
# def consistency_check_with_updates(grouped_left, grouped_right, data_left, data_right, image_dir, interactive=True):
#     """
#     Dynamically detect splits (a single tracking ID in multiple LCA clusters)
#     in Left and Right viewpoints. After each merge, re-map from scratch so we don't
#     repeatedly check old clusters. If user says "No", we rename TIDs (same old logic)
#     but do not re-map immediately.
#     """
#     split_found = False

#     #
#     # 1) Left viewpoint: loop until stable
#     #
#     while True:
#         # Rebuild the tracking ID -> cluster map
#         left_tracking = defaultdict(set)
#         for lca, anns in grouped_left.items():
#             for ann in anns:
#                 left_tracking[ann['tracking_id']].add(lca)

#         any_merge = False

#         # Check each tracking ID for multiple clusters
#         for tid, clusters in left_tracking.items():
#             if len(clusters) > 1:
#                 # We have a "split"
#                 split_found = True
#                 print(f"\n[Left VP] Tracking ID {tid} has split clusters: {clusters}")

#                 # We'll convert to a list so we can attempt merges pairwise
#                 cl_list = list(clusters)
#                 i = 0
#                 while i < len(cl_list) - 1:
#                     c1 = cl_list[i]
#                     c2 = cl_list[i + 1]
#                     print(f"Evaluating clusters {c1} and {c2} for TID {tid}")

#                     did_merge = pairwise_verification_with_update(
#                         grouped_left, c1, c2, data_left, image_dir, interactive, stage="split"
#                     )
#                     if did_merge:
#                         # We performed a merge -> cluster2 is now in c1, or c2 is gone, etc.
#                         any_merge = True
#                         break  # Rebuild everything from scratch
#                     else:
#                         # User said "No", or skip -> rename done, keep going
#                         pass

#                     # Filter out any cluster that no longer exists
#                     cl_list = [c for c in cl_list if c in grouped_left]
#                     i += 1

#             if any_merge:
#                 # We merged something, break this "for tid" loop
#                 break

#         if not any_merge:
#             # No merges in this pass => stable
#             break

#     #
#     # 2) Right viewpoint: same pattern
#     #
#     while True:
#         right_tracking = defaultdict(set)
#         for lca, anns in grouped_right.items():
#             for ann in anns:
#                 right_tracking[ann['tracking_id']].add(lca)

#         any_merge = False

#         for tid, clusters in right_tracking.items():
#             if len(clusters) > 1:
#                 split_found = True
#                 print(f"\n[Right VP] Tracking ID {tid} has split clusters: {clusters}")

#                 cl_list = list(clusters)
#                 i = 0
#                 while i < len(cl_list) - 1:
#                     c1 = cl_list[i]
#                     c2 = cl_list[i + 1]
#                     print(f"Evaluating clusters {c1} and {c2} for TID {tid}")

#                     did_merge = pairwise_verification_with_update(
#                         grouped_right, c1, c2, data_right, image_dir, interactive, stage="split"
#                     )
#                     if did_merge:
#                         any_merge = True
#                         break
#                     else:
#                         pass

#                     cl_list = [c for c in cl_list if c in grouped_right]
#                     i += 1

#             if any_merge:
#                 break

#         if not any_merge:
#             break

#     #
#     # 3) Final summary print
#     #
#     left_tracking = defaultdict(set)
#     for lca, anns in grouped_left.items():
#         for ann in anns:
#             left_tracking[ann['tracking_id']].add(lca)

#     right_tracking = defaultdict(set)
#     for lca, anns in grouped_right.items():
#         for ann in anns:
#             right_tracking[ann['tracking_id']].add(lca)

#     print("\n[Split Stage] Final consistency check (tracking ID -> clusters):")
#     all_tids = set(left_tracking.keys()).union(right_tracking.keys())
#     for tid in all_tids:
#         lcs = left_tracking.get(tid, set())
#         rcs = right_tracking.get(tid, set())
#         print(f"Tracking ID {tid} -> Left clusters: {lcs} | Right clusters: {rcs}")
#         if len(lcs) == len(rcs):
#             print("  Status: Consistent")
#         else:
#             print("  Status: Inconsistent")

#     return split_found

# # =========================
# # Stage 6: Time-Overlap Verification
# # =========================
# def parse_timestamp(ts_str):
#     date_part, time_part = ts_str.split(" ")
#     first_comma = time_part.find(",")
#     if first_comma != -1:
#         time_part = time_part[:first_comma] + "." + time_part[first_comma+1:]
#         time_part = time_part.replace(",", "")
#     fixed = f"{date_part} {time_part}"
#     return datetime.strptime(fixed, "%Y-%m-%d %H:%M:%S.%f")

# def intervals_overlap(start1, end1, start2, end2):
#     return start1 <= end2 and start2 <= end1

# def find_clusters_with_no_time_overlap(grouped_annotations, threshold=timedelta(seconds=1)):
#     cluster_intervals = defaultdict(dict)
#     for cl, anns in grouped_annotations.items():
#         tid_to_times = defaultdict(list)
#         for ann in anns:
#             if 'timestamp' not in ann:
#                 continue
#             ts_str = ann['timestamp']
#             try:
#                 dt = parse_timestamp(ts_str)
#                 tid_to_times[ann['tracking_id']].append(dt)
#             except:
#                 continue
#         for tid, times in tid_to_times.items():
#             times.sort()
#             cluster_intervals[cl][tid] = (times[0], times[-1])

#     clusters = sorted(cluster_intervals.keys())
#     no_overlap = []
#     for i in range(len(clusters)):
#         for j in range(i+1, len(clusters)):
#             c1 = clusters[i]
#             c2 = clusters[j]
#             found = False
#             for t1 in cluster_intervals[c1]:
#                 start1, end1 = cluster_intervals[c1][t1]
#                 for t2 in cluster_intervals[c2]:
#                     start2, end2 = cluster_intervals[c2][t2]
#                     if intervals_overlap(start1, end1, start2, end2):
#                         found = True
#                         break
#                 if found:
#                     break
#             if not found:
#                 no_overlap.append((c1, c2))
#     return no_overlap

# def time_overlap_verification(grouped_annotations, data, image_dir, interactive=True):
#     print("\n[Stage 6] Checking for cluster pairs with NO time overlap...")
#     no_overlap_pairs = find_clusters_with_no_time_overlap(grouped_annotations)
#     if not no_overlap_pairs:
#         print("All clusters have some time overlap.")
#         return

#     print(f"Found {len(no_overlap_pairs)} cluster pair(s) with no time overlap.")
#     for c1, c2 in no_overlap_pairs:
#         print(f"\nCluster pair with NO overlap: {c1} & {c2}")
#         pairwise_verification_with_update(grouped_annotations, c1, c2, data, image_dir, interactive, stage="time")

# # =========================
# # Stage 5: Cluster Equivalence & Individual ID Assignment
# # =========================
# def group_annotations_by_LCA_with_viewpoint(data, viewpoint):
#     grouped = defaultdict(list)
#     for ann in data['annotations']:
#         lca = ann.get('LCA_clustering_id')
#         if lca is not None:
#             grouped[f"{lca}_{viewpoint}"].append(ann)
#     return grouped

# def check_cluster_equivalence(grouped_left, grouped_right):
#     print("\nStep 3: Checking Cluster Equivalence")
#     print("====================================\n")
#     tracking_left = {}
#     tracking_right = {}

#     for cl, anns in grouped_left.items():
#         for ann in anns:
#             tracking_left[ann['tracking_id']] = cl

#     for cl, anns in grouped_right.items():
#         for ann in anns:
#             tracking_right[ann['tracking_id']] = cl

#     equiv = {"left_to_right": defaultdict(set), "right_to_left": defaultdict(set)}

#     for left_cl, anns in grouped_left.items():
#         left_tids = {ann['tracking_id'] for ann in anns}
#         rcs = {tracking_right[tid] for tid in left_tids if tid in tracking_right}
#         equiv["left_to_right"][left_cl] = rcs
#         names = {rc.rsplit('_',1)[0] for rc in rcs}
#         if len(names) > 1:
#             print(f"Cluster {left_cl.rsplit('_',1)[0]} in Left VP spans multiple clusters in Right VP: {names}")
#         else:
#             print(f"Cluster {left_cl.rsplit('_',1)[0]} in Left VP is equivalent to Right VP cluster(s): {names}")

#     for right_cl, anns in grouped_right.items():
#         right_tids = {ann['tracking_id'] for ann in anns}
#         lcs = {tracking_left[tid] for tid in right_tids if tid in tracking_left}
#         equiv["right_to_left"][right_cl] = lcs
#         names = {lc.rsplit('_',1)[0] for lc in lcs}
#         if len(names) > 1:
#             print(f"Cluster {right_cl.rsplit('_',1)[0]} in Right VP spans multiple clusters in Left VP: {names}")
#         else:
#             print(f"Cluster {right_cl.rsplit('_',1)[0]} in Right VP is equivalent to Left VP cluster(s): {names}")

#     print("\nStep 3 Completed: Cluster Equivalence Check Done.\n")
#     return equiv

# def assign_ids_after_equivalence_check(data_left, data_right):
#     grouped_left = group_annotations_by_LCA_with_viewpoint(data_left, 'left')
#     grouped_right = group_annotations_by_LCA_with_viewpoint(data_right, 'right')
#     equiv = check_cluster_equivalence(grouped_left, grouped_right)

#     individual_id = 1
#     processed = set()
#     eq_groups = []

#     # Build initial sets of "equivalent" clusters across viewpoints
#     for left_cl, right_cls in equiv["left_to_right"].items():
#         if right_cls:
#             group = set([left_cl]) | right_cls
#             eq_groups.append(group)

#     # Merge any overlapping sets in eq_groups
#     merged_groups = []
#     while eq_groups:
#         first = eq_groups.pop(0)
#         merged = False
#         for i, grp in enumerate(merged_groups):
#             if first & grp:
#                 merged_groups[i] = grp | first
#                 merged = True
#                 break
#         if not merged:
#             merged_groups.append(first)

#     # Assign IDs to each merged group
#     for group in merged_groups:
#         clusters = group - processed
#         if clusters:
#             views = {cl.split('_')[-1] for cl in clusters}
#             if len(views) > 1:
#                 names = {cl.rsplit('_',1)[0] for cl in clusters}
#                 print(f"Equivalent Clusters {names}. Assigning Individual ID: {individual_id}")
#                 for cl in clusters:
#                     if cl.endswith('_left'):
#                         for ann in grouped_left[cl]:
#                             ann['individual_id'] = individual_id
#                     else:
#                         for ann in grouped_right[cl]:
#                             ann['individual_id'] = individual_id
#                     processed.add(cl)
#                 individual_id += 1
#             else:
#                 # cluster is in one viewpoint only
#                 for cl in clusters:
#                     name = cl.rsplit('_',1)[0]
#                     print(f"Unmatched: {cl} in {'Left' if cl.endswith('_left') else 'Right'} VP. "
#                           f"Assigning Temporary ID: temp_{name}")
#                     if cl.endswith('_left'):
#                         for ann in grouped_left[cl]:
#                             ann['individual_id'] = f"temp_{name}"
#                     else:
#                         for ann in grouped_right[cl]:
#                             ann['individual_id'] = f"temp_{name}"
#                     processed.add(cl)

#     # Handle any leftover clusters not in processed
#     for cl, anns in grouped_left.items():
#         if cl not in processed:
#             name = cl.rsplit('_',1)[0]
#             print(f"Unmatched: Left Cluster {name}. Assigning Temporary ID: temp_{name}")
#             for ann in anns:
#                 ann['individual_id'] = f"temp_{name}"
#             processed.add(cl)

#     for cl, anns in grouped_right.items():
#         if cl not in processed:
#             name = cl.rsplit('_',1)[0]
#             print(f"Unmatched: Right Cluster {name}. Assigning Temporary ID: temp_{name}")
#             for ann in anns:
#                 ann['individual_id'] = f"temp_{name}"
#             processed.add(cl)

#     print("[Stage 5] ID Assignment Completed.")

# # -------------------------
# # Utility JSON I/O Functions
# # -------------------------
# def load_json_data(file_path):
#     with open(file_path, 'r') as f:
#         return json.load(f)

# def save_json_data(data, file_path):
#     with open(file_path, 'w') as f:
#         json.dump(data, f, indent=4)

# # =========================
# # Main Workflow: Chain All Stages Together
# # =========================
# def main():
#     with open("config.yaml", "r") as f:
#         config = yaml.safe_load(f)
#     interactive = config.get("interactive", True)

#     # Stage 1: Merge detection CSV files
#     print("\n--- Stage 1: Merging Detection CSV Files ---")
#     merge_csv_files(
#         config['csv']['detections1'],
#         config['csv']['detections2'],
#         config['csv']['merged_detections']
#     )

#     # Stage 2: Merge CSVs with bounding boxes
#     print("\n--- Stage 2: Merging CSVs with Bounding Boxes ---")
#     merge_csvs_with_bbox(
#         config['csv']['merged_detections'],
#         config['csv']['metadata'],
#         config['csv']['merged_bbox']
#     )

#     # Stage 3: Update JSON annotations with timestamps
#     print("\n--- Stage 3: Updating JSON Annotations with Timestamps ---")
#     update_json_with_timestamp(
#         config['json']['left']['input'],
#         config['csv']['merged_bbox'],
#         config['json']['left']['output']
#     )
#     update_json_with_timestamp(
#         config['json']['right']['input'],
#         config['csv']['merged_bbox'],
#         config['json']['right']['output']
#     )

#     # Load updated JSON data and print initial cluster information.
#     data_left = load_json_data(config['json']['left']['output'])
#     data_right = load_json_data(config['json']['right']['output'])

#     print("\nInitial Left Viewpoint Cluster Summary:")
#     print_cluster_summary(data_left)
#     print_viewpoint_cluster_mapping(data_left, "left")

#     print("\nInitial Right Viewpoint Cluster Summary:")
#     print_cluster_summary(data_right)
#     print_viewpoint_cluster_mapping(data_right, "right")

#     # Stage 4: Split Cluster Verification (Iterative)
#     print("\n--- Stage 4: Split Cluster Verification (Iterative) ---")
#     grouped_left = group_annotations_by_LCA_all(data_left)
#     grouped_right = group_annotations_by_LCA_all(data_right)

#     iteration = 0
#     while True:
#         iteration += 1
#         print(f"\n--- Split Cluster Verification Iteration: {iteration} ---")
#         split_found = consistency_check_with_updates(
#             grouped_left, grouped_right,
#             data_left, data_right,
#             config['image']['directory'],
#             interactive
#         )
#         if not split_found:
#             print("No split clusters found. Proceeding to the next stage.")
#             break

#         # Re-group after changes
#         grouped_left = group_annotations_by_LCA_all(data_left)
#         grouped_right = group_annotations_by_LCA_all(data_right)

#     left_split_file = save_json_with_stage(data_left, config['json']['left']['output'], "split")
#     right_split_file = save_json_with_stage(data_right, config['json']['right']['output'], "split")

#     print("\nAfter Split Cluster Stage - Left VP:")
#     print_cluster_summary(data_left)
#     print("\nAfter Split Cluster Stage - Right VP:")
#     print_cluster_summary(data_right)

#     # Stage 6: Time-Overlap Verification
#     print("\n--- Stage 6: Time-Overlap Verification ---")
#     print("\nLeft viewpoint:")
#     time_overlap_verification(grouped_left, data_left, config['image']['directory'], interactive)

#     print("\nRight viewpoint:")
#     time_overlap_verification(grouped_right, data_right, config['image']['directory'], interactive)

#     left_time_file = save_json_with_stage(data_left, left_split_file, "time")
#     right_time_file = save_json_with_stage(data_right, right_split_file, "time")

#     print("\nAfter Time-Overlap Stage - Left VP:")
#     print_cluster_summary(data_left)
#     print("\nAfter Time-Overlap Stage - Right VP:")
#     print_cluster_summary(data_right)

#     # Stage 5: Cluster Equivalence & Individual ID Assignment
#     print("\n--- Stage 5: Cluster Equivalence and Individual ID Assignment ---")
#     print_viewpoint_cluster_mapping(data_left, "left")
#     print_viewpoint_cluster_mapping(data_right, "right")

#     assign_ids_after_equivalence_check(data_left, data_right)

#     left_final_file = save_json_with_stage(data_left, left_time_file, "cluster_equivalence")
#     right_final_file = save_json_with_stage(data_right, right_time_file, "cluster_equivalence")

#     print("\nFinal Cluster Summary - Left VP:")
#     print_cluster_summary(data_left)
#     print("\nFinal Cluster Summary - Right VP:")
#     print_cluster_summary(data_right)

#     print("\nAll stages completed.")

# if __name__ == "__main__":
#     main()




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
    from IPython.display import display, clear_output
except ImportError:
    widgets = None
    print("ipywidgets not available; falling back to console input for interactive decisions.")

# -------------------------
# Helper function for interactive waiting using ipywidgets.
# -------------------------
def wait_for_dropdown_value(dropdown):
    """Block until the dropdown value is changed from 'Select'."""
    # In a non-blocking environment like Jupyter, true blocking is tricky.
    # This relies on the cell execution waiting.
    # For console, input() is naturally blocking.
    while dropdown.value == 'Select':
        if hasattr(sys, 'stdout'): # Basic check if in an environment that can process events
             plt.pause(0.1) # Allow GUI events to process if possible
        else:
            time.sleep(0.1) # Fallback for simpler environments
    return dropdown.value

# -------------------------
# Helper: Get user decision (using either widgets or input)
# -------------------------
def get_user_decision(prompt="Merge clusters? (Yes/No): ", interactive_mode=True):
    if interactive_mode and widgets is not None:
        dropdown = widgets.Dropdown(
            options=['Select', 'Yes', 'No'],
            value='Select',
            description=prompt,
            style={'description_width': 'initial'},
            layout={'width': 'max-content'} # Try to make description fully visible
        )
        display(dropdown)
        decision = wait_for_dropdown_value(dropdown)
        dropdown.close() # Remove the widget after selection
        # Consider clear_output(wait=True) here if widgets stack up undesirably
        return decision
    else:
        while True:
            decision_input = input(prompt).strip().lower()
            if decision_input.startswith("y"):
                return "Yes"
            elif decision_input.startswith("n"):
                return "No"
            print("Invalid input. Please enter Yes or No.")
            
# -------------------------
# Helper: Get image filename (DEFINITION MOVED HERE OR EARLIER)
# -------------------------
def get_image_filename(images_metadata_list, image_uuid, image_dir_path):
    for img_meta in images_metadata_list:
        if img_meta['uuid'] == image_uuid:
            return os.path.join(image_dir_path, img_meta['file_name'])
    return None


# -------------------------
# Helper: Save JSON file with stage suffix.
# -------------------------
def save_json_with_stage(data, original_filename, stage_suffix, run_identifier=""):
    base, ext = os.path.splitext(original_filename)
    if run_identifier:
        new_filename = f"{base}_{run_identifier}_{stage_suffix}{ext}"
    else:
        new_filename = f"{base}_{stage_suffix}{ext}"
    with open(new_filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved file: {new_filename}")
    return new_filename

# -------------------------
# Helper: Print cluster summary for a JSON data file.
# -------------------------
def print_cluster_summary(data, stage_name=""):
    annotations = data.get("annotations", [])
    total = len(annotations)
    grouped = defaultdict(list)
    for ann in annotations:
        key = ann.get("LCA_clustering_id")
        if key is not None:
            grouped[key].append(ann)
    unique = len(grouped)
    print(f"\n{stage_name} Cluster Summary - Total annotations: {total}, Unique LCA_clustering_ids: {unique}\n")
    for cluster in sorted(grouped.keys()):
        print(f"LCA_clustering_id: {cluster}")
        # Iterate through all annotations in the cluster and print details for each
        for ann in grouped[cluster]:
            ca = ann.get("CA_score", 0)
            ind_id = ann.get("individual_id", "N/A") # Original GT
            final_id = ann.get("final_id", "N/A") # Assigned ID
            print(
                f"  TrkID: {ann.get('tracking_id')}, GT_IndID: {ind_id}, FinalID: {final_id}, "
                f"CA: {ca:.4f}, UUID: {ann.get('uuid')}"
            )
        print() # Print a blank line after each cluster's annotations


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


# =========================
# Stage 1: Merge Detection CSV Files
# =========================
def merge_csv_files(file1_path, file2_path, output_path):
    # Using the simpler version from the second script
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    merged_df = pd.concat([df1, df2], ignore_index=True)
    # Sorting by timestamp is good practice from the first script
    if 'timestamp' in merged_df.columns:
        merged_df = merged_df.sort_values('timestamp')
    merged_df.to_csv(output_path, index=False)
    print(f"[Stage 1] Merged detection CSV files into: {output_path}")
    print(f"[Stage 1] Total rows: {len(merged_df)}")

# =========================
# Stage 2: Merge CSVs with Bounding Boxes
# =========================
def parse_bbox_string_xyxy_s2(bbox_str): # s2 for stage 2 to avoid conflict if others exist
    s = str(bbox_str).strip() # Ensure it's a string
    s = re.sub(r'[\[\]\(\)]', '', s)
    parts = s.split(',')
    try:
        return tuple(int(float(p.strip())) for p in parts) # float then int for robustness
    except Exception as e:
        # print(f"Error parsing bbox (xyxy): {bbox_str} -> {e}")
        return None

def parse_bbox_string_xywh_s2(bbox_str): # s2 for stage 2
    s = str(bbox_str).strip() # Ensure it's a string
    s = re.sub(r'[\[\]\(\)]', '', s)
    parts = s.split(',')
    try:
        coords = [int(float(p.strip())) for p in parts] # float then int
        if len(coords) != 4:
            return None
        x1, y1, w, h = coords
        return (x1, y1, x1 + w, y1 + h)
    except Exception as e:
        # print(f"Error parsing bbox (xywh): {bbox_str} -> {e}")
        return None

def merge_csvs_with_bbox(timestamps_csv, metadata_csv, output_csv):
    # Using the simpler version from the second script, with parsing from first script
    dict_ts = {}
    with open(timestamps_csv, 'r', newline='', encoding='utf-8') as f1:
        reader1 = csv.DictReader(f1)
        ts_fieldnames = reader1.fieldnames or []
        for row in reader1:
            ts_filename = row.get('frame_name')
            if not ts_filename: continue
            ts_bbox_str = row.get('bounding_box')
            if not ts_bbox_str: continue
            ts_bbox = parse_bbox_string_xyxy_s2(ts_bbox_str) # Use robust parser
            if ts_bbox is None: continue
            key = (ts_filename, ts_bbox)
            dict_ts[key] = row

    merged_rows = []
    combined_fieldnames = set(ts_fieldnames)
    with open(metadata_csv, 'r', newline='', encoding='utf-8') as f2:
        reader2 = csv.DictReader(f2)
        metadata_fieldnames = reader2.fieldnames or []
        for col in metadata_fieldnames: combined_fieldnames.add(col)
        for row in reader2:
            md_filename = row.get('file_name')
            if not md_filename: continue
            md_bbox_str = row.get('bbox')
            if not md_bbox_str: continue
            md_bbox = parse_bbox_string_xywh_s2(md_bbox_str) # Use robust parser
            if md_bbox is None: continue
            key = (md_filename, md_bbox)
            if key in dict_ts:
                ts_row = dict_ts[key]
                merged_dict = {**ts_row, **row} # Simpler merge
                merged_rows.append(merged_dict)

    combined_fieldnames = list(combined_fieldnames)
    with open(output_csv, 'w', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, fieldnames=combined_fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(merged_rows)

    print(f"[Stage 2] Merged CSV with bounding boxes written to: {output_csv}")
    print(f"[Stage 2] Total merged rows: {len(merged_rows)}")


# =========================
# Stage 3: Update JSON Annotations with Timestamps
# =========================
# Using the more robust parsing from the first script for make_comparable_dicts
def parse_bbox_for_json_update(bbox_str): # From first script's Stage 3
    try:
        return list(ast.literal_eval(bbox_str))
    except:
        return None

def make_comparable_dict_from_csv_s3(row, file_to_uuid): # s3 from script 1's stage 3 logic
    def safe_int(x):
        try: return int(str(x).strip())
        except: return None
    def safe_float(x):
        try: return float(str(x).strip())
        except: return None

    tracking_id = safe_int(row.get("tracking_id", ""))
    confidence = safe_float(row.get("confidence", ""))
    detection_class = safe_int(row.get("detection_class", ""))
    CA_score = safe_float(row.get("CA_score", ""))
    # Important: individual_id from CSV is often GT. Preserve it.
    individual_id = safe_int(row.get("individual_id", ""))
    species = row.get("species", "").strip()
    bbox = parse_bbox_for_json_update(row.get("bbox", ""))
    image_uuid = file_to_uuid.get(row.get("file_name", "").strip(), None)
    timestamp = row.get("timestamp", "").strip()

    return {
        "image_uuid": image_uuid, "tracking_id": tracking_id,
        "confidence": confidence, "detection_class": detection_class,
        "species": species, "bbox": bbox, "CA_score": CA_score,
        "individual_id": individual_id, # Make sure this is part of the comparison key
    }, timestamp

def make_comparable_dict_from_json_s3(ann): # s3 from script 1's stage 3 logic
    bbox_list = list(ann["bbox"]) if ann.get("bbox") else None
    # Ensure all relevant fields from CSV are included for comparison
    return {
        "image_uuid": ann.get("image_uuid"), "tracking_id": ann.get("tracking_id"),
        "confidence": ann.get("confidence"), "detection_class": ann.get("detection_class"),
        "species": ann.get("species"), "bbox": bbox_list, "CA_score": ann.get("CA_score"),
        # individual_id from JSON is also important
        "individual_id": ann.get("individual_id"),
    }

def update_json_with_timestamp(json_input, csv_input, json_output): # Stage 3 logic
    with open(json_input, "r") as f: data = json.load(f)
    file_to_uuid = {img["file_name"]: img["uuid"] for img in data["images"]}
    csv_common_list = []
    with open(csv_input, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            comp_fields, ts = make_comparable_dict_from_csv_s3(row, file_to_uuid)
            if comp_fields["image_uuid"] is not None: # Must have image_uuid to match
                csv_common_list.append((comp_fields, ts))

    updated_count = 0
    for (csv_fields_to_match, csv_timestamp) in csv_common_list:
        for ann in data["annotations"]:
            ann_fields_to_match = make_comparable_dict_from_json_s3(ann)
            # Strict comparison of all specified fields
            match = all(ann_fields_to_match.get(k) == csv_fields_to_match.get(k) for k in csv_fields_to_match)
            if match:
                ann["timestamp"] = csv_timestamp
                updated_count += 1
                break # Found match for this CSV row, move to next
    with open(json_output, "w") as f: json.dump(data, f, indent=4)
    print(f"[Stage 3] Updated JSON written to: {json_output}")
    print(f"[Stage 3] Number of annotations updated with timestamp: {updated_count}")


def group_annotations_by_LCA_all(data): # DEFINITION SHOULD BE HERE OR EARLIER
    annotations = data['annotations']
    grouped = defaultdict(list)
    for ann in annotations:
        lca = ann.get('LCA_clustering_id')
        if lca is not None:
            grouped[lca].append(ann)
    return grouped
# =========================
# Stage 4 & 6: Interactive Verification Core Logic
# =========================
def get_cluster_best_ann_for_display(annotations_list):
    if not annotations_list: return None
    return max(annotations_list, key=lambda x: x.get("CA_score", 0.0))

def _update_cluster_merge_deterministic(grouped_annotations, source_cluster_id, target_cluster_id):
    """ Merges source_cluster into target_cluster. Assumes target_cluster_id is the anchor. """
    if source_cluster_id not in grouped_annotations or target_cluster_id not in grouped_annotations: return
    if source_cluster_id == target_cluster_id: return

    grouped_annotations[target_cluster_id].extend(grouped_annotations[source_cluster_id])
    for ann in grouped_annotations[source_cluster_id]:
        ann['LCA_clustering_id'] = target_cluster_id
    del grouped_annotations[source_cluster_id]
    print(f"    Merged cluster {source_cluster_id} into {target_cluster_id}.")


def _update_split_no_merge_deterministic(grouped_annotations, anchor_cluster_id, other_cluster_id):
    """ Handles TIDs for no-merge in split stage, using anchor and other. """
    if anchor_cluster_id not in grouped_annotations or other_cluster_id not in grouped_annotations: return

    anchor_tids = {ann['tracking_id'] for ann in grouped_annotations[anchor_cluster_id]}
    renamed_count = 0
    for ann_in_other in grouped_annotations[other_cluster_id]:
        if ann_in_other['tracking_id'] in anchor_tids:
            original_tid = ann_in_other['tracking_id']
            ann_in_other['tracking_id'] = f"{str(original_tid)}_new"
            renamed_count +=1
    if renamed_count > 0:
        print(f"    No merge: Updated {renamed_count} conflicting tracking_ids in cluster {other_cluster_id} (suffix '_new').")
    else:
        print(f"    No merge: No conflicting tracking_ids found between {anchor_cluster_id} and {other_cluster_id}.")


def pairwise_verification_interactive_deterministic(
    grouped_annotations, cluster1_id, cluster2_id,
    data_context_for_display, # This will be data_left or data_right
    image_dir_path, interactive_mode, stage
):
    if cluster1_id not in grouped_annotations or cluster2_id not in grouped_annotations or \
       not grouped_annotations[cluster1_id] or not grouped_annotations[cluster2_id]:
        print(f"  [{stage.capitalize()}] Skipping: {cluster1_id} or {cluster2_id} is missing/empty.")
        return False # No merge happened

    best_ann1 = get_cluster_best_ann_for_display(grouped_annotations[cluster1_id])
    best_ann2 = get_cluster_best_ann_for_display(grouped_annotations[cluster2_id])

    if not best_ann1 or not best_ann2:
        print(f"  [{stage.capitalize()}] Error: Could not get best annotations for display. Skipping.")
        return False

    # Display images using the helper from the second script (modified for pair)
    # (Assuming display_image_pair_for_decision is defined elsewhere or inline it)
    print(f"\n  [{stage.capitalize()} Stage] User Verification: Clusters '{cluster1_id}' and '{cluster2_id}'")
    # (Image display logic would go here using display_image_pair_for_decision if defined)
    if image_dir_path:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for i, ann_ref in enumerate([best_ann1, best_ann2]):
            ax = axes[i]
            img_uuid = ann_ref['image_uuid']
            file_path = get_image_filename(data_context_for_display['images'], img_uuid, image_dir_path)
            if file_path and os.path.exists(file_path):
                try:
                    image = Image.open(file_path)
                    x, y, w, h = ann_ref['bbox']
                    cropped = image.crop((x, y, x + w, y + h))
                    ax.imshow(cropped)
                    ax.set_title(f"Cls: {ann_ref['LCA_clustering_id']}\nTrkID: {ann_ref['tracking_id']}\nCA: {ann_ref.get('CA_score',0):.3f}")
                    ax.axis('off')
                except Exception as e: ax.text(0.5,0.5,f"Err: {e}", ha='center'); ax.axis('off')
            else: ax.text(0.5,0.5,"Img N/F", ha='center'); ax.axis('off')
        plt.tight_layout()
        if widgets and interactive_mode: display(fig)
        else: plt.show(block=False) # Show non-blocking for console
    else:
        print("    (Image display skipped as image_dir is not configured)")


    decision = get_user_decision(prompt=f"    Merge {cluster1_id} & {cluster2_id}? (Yes/No): ", interactive_mode=interactive_mode)
    if widgets and interactive_mode: clear_output(wait=True) # Clean up Jupyter display

    anchor_id, other_id = sorted([cluster1_id, cluster2_id]) # Deterministic anchor/other

    if decision == "Yes":
        print(f"    User chose to MERGE. Anchor: {anchor_id}, Other: {other_id}")
        _update_cluster_merge_deterministic(grouped_annotations, other_id, anchor_id)
        return True # Merge happened
    else: # Decision == "No"
        print(f"    User chose NOT to merge. Anchor: {anchor_id}, Other: {other_id}")
        if stage == 'split':
            _update_split_no_merge_deterministic(grouped_annotations, anchor_id, other_id)
        # For 'time' stage, no-merge means no specific action on TIDs (consistent)
        return False # No merge happened

def consistency_check_interactive_deterministic(
    grouped_annotations_view, data_view, # e.g., grouped_left, data_left
    viewpoint_name, image_dir_path, interactive_mode, stage="split"
):
    print(f"\n--- {viewpoint_name} Viewpoint {stage.capitalize()} Consistency Check ---")
    overall_changes_made = False
    while True: # Loop for this viewpoint until no more merges in a pass
        tid_to_lca_map = defaultdict(set)
        for lca_id, anns_list in grouped_annotations_view.items():
            for ann_item in anns_list:
                tid_to_lca_map[ann_item['tracking_id']].add(lca_id)
        
        made_merge_this_pass = False
        conflicting_tids = {tid: lcas for tid, lcas in tid_to_lca_map.items() if len(lcas) > 1}

        if not conflicting_tids:
            print(f"  No tracking ID splits found in {viewpoint_name} viewpoint.")
            break

        for tid, lca_set in conflicting_tids.items():
            print(f"  TID {tid} found in multiple LCAs: {lca_set} in {viewpoint_name}")
            sorted_lca_list = sorted(list(lca_set))
            
            # Check pairs. If a merge happens, restart the pass for this viewpoint.
            # This is complex because merging c1,c2 changes grouped_annotations_view
            # and can invalidate further items in sorted_lca_list or conflicting_tids
            
            # Simplified: Process one conflict pair at a time and if merge, restart
            c1_id, c2_id = sorted_lca_list[0], sorted_lca_list[1]

            if pairwise_verification_interactive_deterministic(
                grouped_annotations_view, c1_id, c2_id,
                data_view, image_dir_path, interactive_mode, stage
            ):
                overall_changes_made = True
                made_merge_this_pass = True
                break # Break from for_tid loop to restart while True pass
        
        if not made_merge_this_pass:
            print(f"  No merges made in this pass for {viewpoint_name}. Consistency check for this viewpoint stable.")
            break # Viewpoint is stable
        else:
            print(f"  Merge occurred in {viewpoint_name}, re-evaluating consistency...")
    return overall_changes_made


# =========================
# Stage 6: Time-Overlap Verification Helpers
# (These should be defined before time_overlap_verification_interactive_deterministic)
# =========================
def parse_timestamp(ts_str): # Helper for find_clusters_with_no_time_overlap
    """ Parses a timestamp string into a datetime object. Handles variations with commas for milliseconds. """
    if not isinstance(ts_str, str): # Add a check if ts_str is not a string
        # print(f"Warning: Timestamp provided is not a string: {ts_str}. Skipping parse.")
        raise ValueError("Timestamp input must be a string.")

    try:
        # Attempt direct parsing first for standard formats
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        try:
            # Handle cases where milliseconds might be separated by a comma or have varying lengths
            date_part, time_part = ts_str.split(" ", 1)
            if ',' in time_part:
                time_part = time_part.replace(',', '.', 1) # Replace only the first comma for milliseconds
            
            # Normalize fractional seconds to 6 digits if possible, or truncate
            if '.' in time_part:
                main_time_part, fractional_part = time_part.split('.', 1)
                fractional_part = fractional_part.ljust(6, '0')[:6] # Ensure 6 digits, truncate if longer
                time_part = f"{main_time_part}.{fractional_part}"
            else: # No fractional part, add .000000
                 time_part = f"{time_part}.000000"

            fixed_ts_str = f"{date_part} {time_part}"
            return datetime.strptime(fixed_ts_str, "%Y-%m-%d %H:%M:%S.%f")
        except Exception as e:
            # print(f"Error parsing timestamp string '{ts_str}': {e}")
            raise # Re-raise the exception if parsing still fails

def intervals_overlap(start1, end1, start2, end2): # Helper for find_clusters_with_no_time_overlap
    """ Checks if two time intervals [start1, end1] and [start2, end2] overlap. """
    return start1 <= end2 and start2 <= end1

def find_clusters_with_no_time_overlap(grouped_annotations, threshold_timedelta=timedelta(seconds=1)):
    """
    Identifies pairs of clusters that have no temporal overlap in their annotations.
    Args:
        grouped_annotations (dict): Keys are LCA_clustering_ids, values are lists of annotation dicts.
        threshold_timedelta (timedelta): A small timedelta, not actively used in this version for strict no-overlap.
                                          Original second script had a 'threshold' param here, keeping for signature.
    Returns:
        list: A list of tuples, where each tuple contains two LCA_clustering_ids that do not overlap in time.
    """
    cluster_intervals = defaultdict(dict) # {lca_id: {tracking_id: (min_time, max_time)}}
    
    for lca_id, anns_list in grouped_annotations.items():
        tid_to_times_map = defaultdict(list)
        for ann_item in anns_list:
            if 'timestamp' not in ann_item or not ann_item['timestamp']: # Check for missing or empty timestamp
                # print(f"Warning: Annotation {ann_item.get('uuid', 'Unknown UUID')} in LCA {lca_id} missing timestamp. Skipping.")
                continue
            
            timestamp_str = ann_item['timestamp']
            try:
                datetime_obj = parse_timestamp(timestamp_str)
                tid_to_times_map[ann_item.get('tracking_id', 'UnknownTID')].append(datetime_obj)
            except ValueError as e: # Catch parsing errors from parse_timestamp
                # print(f"Warning: Could not parse timestamp '{timestamp_str}' for ann {ann_item.get('uuid', 'Unknown UUID')} in LCA {lca_id}. Error: {e}")
                continue # Skip this annotation's timestamp
            except Exception as e_gen: # Catch any other unexpected error during parsing
                # print(f"Unexpected error parsing timestamp '{timestamp_str}': {e_gen}")
                continue


        for tid, times_list in tid_to_times_map.items():
            if times_list: # Ensure there are actual time objects
                times_list.sort()
                cluster_intervals[lca_id][tid] = (times_list[0], times_list[-1])
            # else:
                # print(f"Warning: Tracking ID {tid} in LCA {lca_id} had no valid timestamps after parsing.")

    # Sort cluster IDs for consistent pairing
    sorted_lca_ids = sorted(cluster_intervals.keys())
    non_overlapping_pairs = []

    for i in range(len(sorted_lca_ids)):
        for j in range(i + 1, len(sorted_lca_ids)):
            lca_id1 = sorted_lca_ids[i]
            lca_id2 = sorted_lca_ids[j]

            # Ensure both clusters still have interval data (could be empty if all timestamps failed)
            if not cluster_intervals[lca_id1] or not cluster_intervals[lca_id2]:
                continue

            overlap_found_between_lcas = False
            # Check if any tracking ID interval in lca_id1 overlaps with any in lca_id2
            for tid1_intervals in cluster_intervals[lca_id1].values():
                start1, end1 = tid1_intervals
                for tid2_intervals in cluster_intervals[lca_id2].values():
                    start2, end2 = tid2_intervals
                    if intervals_overlap(start1, end1, start2, end2):
                        overlap_found_between_lcas = True
                        break # Found overlap for this tid1, move to next tid1 or lca pair
                if overlap_found_between_lcas:
                    break # Found overlap for this lca_id1, move to next lca pair
            
            if not overlap_found_between_lcas:
                non_overlapping_pairs.append((lca_id1, lca_id2))
                
    return non_overlapping_pairs

# =========================
# Stage 6: Time-Overlap Verification Function (Interactive)
# =========================
def time_overlap_verification_interactive_deterministic(
    grouped_annotations_view, data_view, viewpoint_name,
    image_dir_path, interactive_mode
):
    print(f"\n--- {viewpoint_name} Viewpoint Time-Overlap Verification ---")
    # Now find_clusters_with_no_time_overlap is defined
    no_overlap_pairs = find_clusters_with_no_time_overlap(grouped_annotations_view) 

    if not no_overlap_pairs:
        print(f"  All clusters in {viewpoint_name} have some time overlap or too few clusters to check.")
        return False # No changes made

    print(f"  Found {len(no_overlap_pairs)} cluster pair(s) in {viewpoint_name} with NO time overlap.")
    any_merges_in_time_stage = False
    for c1_id, c2_id in no_overlap_pairs:
        # Ensure clusters still exist (might have been merged in a previous iteration of this loop)
        if c1_id not in grouped_annotations_view or c2_id not in grouped_annotations_view:
            # print(f"    Skipping pair ({c1_id}, {c2_id}): one or both clusters no longer exist.")
            continue
        print(f"  Verifying non-overlapping pair: {c1_id} & {c2_id}")
        if pairwise_verification_interactive_deterministic( # This is your core interactive function
            grouped_annotations_view, c1_id, c2_id,
            data_view, image_dir_path, interactive_mode, stage="time" # stage="time"
        ):
            any_merges_in_time_stage = True
            # If a merge happens, grouped_annotations_view is modified.
            # The list no_overlap_pairs is static. Continuing the loop might lead to
            # attempting to verify a cluster that was just deleted.
            # A more robust approach would be to re-calculate no_overlap_pairs after each merge,
            # but for simplicity here, we'll continue and rely on the checks within
            # pairwise_verification_interactive_deterministic.
            print(f"    Merge occurred for {c1_id}, {c2_id}. Grouped annotations updated.")
    
    if any_merges_in_time_stage:
        print(f"  Merges were made in {viewpoint_name} during time-overlap verification.")
    else:
        print(f"  No merges made in {viewpoint_name} during time-overlap verification based on user decisions.")
    return any_merges_in_time_stage

def group_annotations_by_LCA_with_viewpoint(data, viewpoint): # Ensure this definition exists and is placed here or earlier
    """Groups annotations by LCA_clustering_id, appending the viewpoint to the key."""
    grouped = defaultdict(list)
    for ann in data.get('annotations', []): # Added .get for safety
        lca = ann.get('LCA_clustering_id')
        if lca is not None:
            grouped[f"{lca}_{viewpoint}"].append(ann)
    return grouped

# =========================
# Stage 5: Cluster Equivalence & Individual ID Assignment (Logic from Script 1)
# =========================
# group_annotations_by_LCA_with_viewpoint (already defined for Script 1 logic)

# check_cluster_equivalence (from Script 1 logic)
def check_cluster_equivalence_s1(grouped_left_wv, grouped_right_wv):
    # This is the version from the first script (renamed to avoid conflict if needed)
    # print("\n[Stage 5] Checking Cluster Equivalence (Script 1 logic)")
    tid_to_left  = {ann['tracking_id']: cl_key for cl_key, anns in grouped_left_wv.items() for ann in anns}
    tid_to_right = {ann['tracking_id']: cl_key for cl_key, anns in grouped_right_wv.items() for ann in anns}

    # Summary (optional, can be verbose)
    # ... (summary printing as in script 1) ...

    tid_to_clusters_map = defaultdict(set)
    for cl_key, anns in grouped_left_wv.items():
        for ann in anns: tid_to_clusters_map[ann['tracking_id']].add(cl_key)
    for cl_key, anns in grouped_right_wv.items():
        for ann in anns: tid_to_clusters_map[ann['tracking_id']].add(cl_key)

    all_cluster_keys_in_views = list(grouped_left_wv.keys()) + list(grouped_right_wv.keys())
    adj_list = {cl_key: set() for cl_key in all_cluster_keys_in_views}

    for common_tid_clusters in tid_to_clusters_map.values():
        cluster_list = list(common_tid_clusters)
        for i in range(len(cluster_list)):
            for j in range(i + 1, len(cluster_list)):
                c1, c2 = cluster_list[i], cluster_list[j]
                if c1 in adj_list and c2 in adj_list : # Ensure keys exist
                    adj_list[c1].add(c2)
                    adj_list[c2].add(c1)
    
    visited_nodes = set()
    equivalence_sets = []
    for cl_key_node in all_cluster_keys_in_views:
        if cl_key_node not in visited_nodes and cl_key_node in adj_list:
            current_component = set()
            component_stack = [cl_key_node]
            while component_stack:
                curr = component_stack.pop()
                if curr not in visited_nodes:
                    visited_nodes.add(curr)
                    current_component.add(curr)
                    if curr in adj_list: # curr node might not have any edges if isolated after splits
                         component_stack.extend(adj_list[curr] - visited_nodes)
            if current_component:
                equivalence_sets.append(current_component)
    # print(f"[Stage 5] Found {len(equivalence_sets)} equivalence sets.")
    return equivalence_sets

# generate_new_lca_id (from Script 1)
def generate_new_lca_id_s1(base_lca_id_str, existing_lca_ids_in_viewpoint_data_set):
    base_lca_id_str = str(base_lca_id_str)
    max_suffix = 0
    pattern = re.compile(rf"^{re.escape(base_lca_id_str)}_split(\d+)$")
    for existing_id_str in existing_lca_ids_in_viewpoint_data_set:
        match = pattern.match(str(existing_id_str))
        if match: max_suffix = max(max_suffix, int(match.group(1)))
    new_suffix = max_suffix + 1
    return f"{base_lca_id_str}_split{new_suffix}"

# split_conflicting_clusters_iteratively (from Script 1 logic)
def split_conflicting_clusters_iteratively_s1(grouped_left_wv, grouped_right_wv, data_left, data_right):
    overall_splits_made = False
    max_outer_loops = 5 # Safety break

    for outer_loop_count in range(max_outer_loops):
        # print(f"  [Conflict Resolution by Splitting] Outer Loop: {outer_loop_count + 1}")
        split_made_in_this_pass = False

        # --- Right to Left Conflict Splitting ---
        all_current_right_lca_ids = {str(ann['LCA_clustering_id']) for ann in data_right['annotations'] if 'LCA_clustering_id' in ann}
        
        # Iterate over a copy of keys as grouped_right_wv can be modified
        for r_cluster_key in list(grouped_right_wv.keys()): 
            if r_cluster_key not in grouped_right_wv or not grouped_right_wv[r_cluster_key]:
                if r_cluster_key in grouped_right_wv: del grouped_right_wv[r_cluster_key] # Clean up empty
                continue

            original_r_anns_list = list(grouped_right_wv[r_cluster_key]) # Copy of annotations
            r_cluster_tracking_ids = {ann['tracking_id'] for ann in original_r_anns_list}
            # All annotations in a group should have the same LCA ID (without _right/_left yet)
            r_cluster_base_lca_id = str(original_r_anns_list[0]['LCA_clustering_id']) 

            # Which Left LCA_clustering_ids do these TIDs map to?
            tid_to_left_lca_map = defaultdict(set)
            mapped_left_lca_ids = set()
            for l_cluster_key_iter, l_anns_list_iter in grouped_left_wv.items():
                if not l_anns_list_iter: continue
                current_l_base_lca_id = str(l_anns_list_iter[0]['LCA_clustering_id'])
                for l_ann_iter in l_anns_list_iter:
                    if l_ann_iter['tracking_id'] in r_cluster_tracking_ids:
                        tid_to_left_lca_map[l_ann_iter['tracking_id']].add(current_l_base_lca_id)
                        mapped_left_lca_ids.add(current_l_base_lca_id)
            
            if len(mapped_left_lca_ids) > 1: # Conflict! Right cluster maps to multiple Left LCAs
                print(f"    Conflict: Right Cluster {r_cluster_key} (Base LCA: {r_cluster_base_lca_id}) maps to multiple Left LCAs: {mapped_left_lca_ids}")
                split_made_in_this_pass = True
                overall_splits_made = True
                
                newly_created_split_parts = defaultdict(list) # {new_r_cluster_key_with_view: [anns]}
                moved_ann_uuids_this_conflict = set()

                # For each target Left LCA, create a new split part in Right
                for target_left_lca_id_str in sorted(list(mapped_left_lca_ids)):
                    # Identify TIDs in original_r_anns_list that connect to this target_left_lca_id_str
                    tids_for_this_specific_connection = {
                        tid for tid, mapped_lcas in tid_to_left_lca_map.items() if target_left_lca_id_str in mapped_lcas
                    }
                    if not tids_for_this_specific_connection: continue

                    new_r_lca_id_for_this_split = generate_new_lca_id_s1(r_cluster_base_lca_id, all_current_right_lca_ids)
                    all_current_right_lca_ids.add(new_r_lca_id_for_this_split) # Add to set for next generation
                    new_r_cluster_key_for_this_split = f"{new_r_lca_id_for_this_split}_right"
                    print(f"      Defining new Right split part: {new_r_cluster_key_for_this_split} (for TIDs linking to Left LCA {target_left_lca_id_str})")

                    for r_ann in original_r_anns_list: # Iterate original annotations
                        if r_ann['tracking_id'] in tids_for_this_specific_connection and \
                           r_ann['uuid'] not in moved_ann_uuids_this_conflict:
                            
                            ann_copy_for_split = r_ann.copy()
                            ann_copy_for_split['LCA_clustering_id'] = new_r_lca_id_for_this_split # Update LCA ID
                            newly_created_split_parts[new_r_cluster_key_for_this_split].append(ann_copy_for_split)
                            moved_ann_uuids_this_conflict.add(r_ann['uuid'])
                
                # Update the original r_cluster_key in grouped_right_wv
                remaining_r_anns = [ann for ann in original_r_anns_list if ann['uuid'] not in moved_ann_uuids_this_conflict]
                if not remaining_r_anns:
                    if r_cluster_key in grouped_right_wv: del grouped_right_wv[r_cluster_key]
                else:
                    grouped_right_wv[r_cluster_key] = remaining_r_anns
                
                # Add new split parts to grouped_right_wv
                for new_key, new_anns in newly_created_split_parts.items():
                    if new_anns: grouped_right_wv[new_key] = new_anns

                # Update data_right['annotations'] to reflect all LCA_clustering_id changes
                updated_full_data_right_annotations = []
                for anns_in_cluster in grouped_right_wv.values(): 
                    updated_full_data_right_annotations.extend(anns_in_cluster)
                data_right['annotations'] = updated_full_data_right_annotations
                # Crucial: A split was made, restart this outer loop for stability
                # by breaking from current for-loop (r_cluster_key) and letting outer_loop_count continue
                break 
        
        if split_made_in_this_pass: continue # Restart outer loop if a R->L split occurred

        # --- Left to Right Conflict Splitting (Symmetric) ---
        all_current_left_lca_ids = {str(ann['LCA_clustering_id']) for ann in data_left['annotations'] if 'LCA_clustering_id' in ann}
        for l_cluster_key in list(grouped_left_wv.keys()):
            if l_cluster_key not in grouped_left_wv or not grouped_left_wv[l_cluster_key]:
                if l_cluster_key in grouped_left_wv: del grouped_left_wv[l_cluster_key]
                continue
            original_l_anns_list = list(grouped_left_wv[l_cluster_key])
            l_cluster_tracking_ids = {ann['tracking_id'] for ann in original_l_anns_list}
            l_cluster_base_lca_id = str(original_l_anns_list[0]['LCA_clustering_id'])
            tid_to_right_lca_map = defaultdict(set)
            mapped_right_lca_ids = set()
            for r_cluster_key_iter, r_anns_list_iter in grouped_right_wv.items():
                if not r_anns_list_iter: continue
                current_r_base_lca_id = str(r_anns_list_iter[0]['LCA_clustering_id'])
                for r_ann_iter in r_anns_list_iter:
                    if r_ann_iter['tracking_id'] in l_cluster_tracking_ids:
                        tid_to_right_lca_map[r_ann_iter['tracking_id']].add(current_r_base_lca_id)
                        mapped_right_lca_ids.add(current_r_base_lca_id)
            
            if len(mapped_right_lca_ids) > 1:
                print(f"    Conflict: Left Cluster {l_cluster_key} (Base LCA: {l_cluster_base_lca_id}) maps to multiple Right LCAs: {mapped_right_lca_ids}")
                split_made_in_this_pass = True
                overall_splits_made = True
                newly_created_split_parts = defaultdict(list)
                moved_ann_uuids_this_conflict = set()
                for target_right_lca_id_str in sorted(list(mapped_right_lca_ids)):
                    tids_for_this_specific_connection = {
                        tid for tid, mapped_lcas in tid_to_right_lca_map.items() if target_right_lca_id_str in mapped_lcas
                    }
                    if not tids_for_this_specific_connection: continue
                    new_l_lca_id_for_this_split = generate_new_lca_id_s1(l_cluster_base_lca_id, all_current_left_lca_ids)
                    all_current_left_lca_ids.add(new_l_lca_id_for_this_split)
                    new_l_cluster_key_for_this_split = f"{new_l_lca_id_for_this_split}_left"
                    print(f"      Defining new Left split part: {new_l_cluster_key_for_this_split} (for TIDs linking to Right LCA {target_right_lca_id_str})")
                    for l_ann in original_l_anns_list:
                        if l_ann['tracking_id'] in tids_for_this_specific_connection and \
                           l_ann['uuid'] not in moved_ann_uuids_this_conflict:
                            ann_copy_for_split = l_ann.copy()
                            ann_copy_for_split['LCA_clustering_id'] = new_l_lca_id_for_this_split
                            newly_created_split_parts[new_l_cluster_key_for_this_split].append(ann_copy_for_split)
                            moved_ann_uuids_this_conflict.add(l_ann['uuid'])
                remaining_l_anns = [ann for ann in original_l_anns_list if ann['uuid'] not in moved_ann_uuids_this_conflict]
                if not remaining_l_anns:
                    if l_cluster_key in grouped_left_wv: del grouped_left_wv[l_cluster_key]
                else:
                    grouped_left_wv[l_cluster_key] = remaining_l_anns
                for new_key, new_anns in newly_created_split_parts.items():
                    if new_anns: grouped_left_wv[new_key] = new_anns
                updated_full_data_left_annotations = []
                for anns_in_cluster in grouped_left_wv.values():
                    updated_full_data_left_annotations.extend(anns_in_cluster)
                data_left['annotations'] = updated_full_data_left_annotations
                break # Restart outer loop for L->R split
        
        if not split_made_in_this_pass: # No splits in R->L pass AND no splits in L->R pass
            print(f"  [Conflict Resolution by Splitting] Stable after {outer_loop_count + 1} outer loop(s).")
            return overall_splits_made 

    print(f"  [Conflict Resolution by Splitting] Reached max outer loops ({max_outer_loops}).")
    return overall_splits_made


# assign_ids_after_equivalence_check (from Script 1 logic - assigns 'final_id')
def assign_ids_after_equivalence_check_s1(data_left, data_right, eq_sets_from_s1_check):
    print("\n[Stage 5] Assigning Final IDs (Script 1 logic)")
    assigned_final_ids_map = {}  # cluster_key_with_viewpoint -> final_id
    next_available_numeric_id = 1

    # 1. Assign permanent numeric IDs to groups spanning both viewpoints
    for eq_group in eq_sets_from_s1_check:
        viewpoints_in_group = {cl_key.rsplit("_", 1)[1] for cl_key in eq_group} # e.g., {'left', 'right'}
        if "left" in viewpoints_in_group and "right" in viewpoints_in_group:
            # This group is a confirmed cross-viewpoint individual
            base_lca_ids_in_group = {cl_key.rsplit('_',1)[0] for cl_key in eq_group}
            print(f"  Equivalent Group {base_lca_ids_in_group} (spans L/R). Assigning Final ID: {next_available_numeric_id}")
            for cl_key in eq_group:
                assigned_final_ids_map[cl_key] = str(next_available_numeric_id)
            next_available_numeric_id += 1
    
    # 2. Assign IDs to remaining (single-viewpoint) clusters
    # Left viewpoint clusters get new permanent numeric IDs if not already processed
    all_left_cluster_keys = {f"{ann['LCA_clustering_id']}_left" for ann in data_left['annotations'] if 'LCA_clustering_id' in ann}
    for l_cl_key in sorted(list(all_left_cluster_keys)):
        if l_cl_key not in assigned_final_ids_map:
            base_lca = l_cl_key.rsplit("_", 1)[0]
            print(f"  Unmatched Left Cluster {base_lca}. Assigning New Final ID: {next_available_numeric_id}")
            assigned_final_ids_map[l_cl_key] = str(next_available_numeric_id)
            next_available_numeric_id +=1 # Increment for each new distinct individual

    # Right viewpoint clusters (not part of a L/R group) get temporary IDs
    all_right_cluster_keys = {f"{ann['LCA_clustering_id']}_right" for ann in data_right['annotations'] if 'LCA_clustering_id' in ann}
    for r_cl_key in sorted(list(all_right_cluster_keys)):
        if r_cl_key not in assigned_final_ids_map:
            base_lca = r_cl_key.rsplit("_", 1)[0]
            temp_id = f"temp_{base_lca}_R"
            print(f"  Unmatched Right Cluster {base_lca}. Assigning Temporary Final ID: {temp_id}")
            assigned_final_ids_map[r_cl_key] = temp_id

    # Annotate the JSON data with the 'final_id'
    for ann in data_left['annotations']:
        if 'LCA_clustering_id' in ann:
            key = f"{ann['LCA_clustering_id']}_left"
            ann['final_id'] = assigned_final_ids_map.get(key, "None_L_Err")
    for ann in data_right['annotations']:
        if 'LCA_clustering_id' in ann:
            key = f"{ann['LCA_clustering_id']}_right"
            ann['final_id'] = assigned_final_ids_map.get(key, "None_R_Err")
    print("[Stage 5] Final ID assignment completed.")


# =========================
# Utility JSON I/O Functions (from Script 2)
# =========================
def load_json_data(file_path):
    with open(file_path, 'r') as f: return json.load(f)

def save_json_data(data, file_path): # (already defined as save_json_with_stage)
    with open(file_path, 'w') as f: json.dump(data, f, indent=4)


# =========================
# Main Workflow: Chain All Stages Together
# =========================
def main():
    if len(sys.argv) < 2:
        print("Usage: python interactive_script.py <config.yaml>")
        sys.exit(1)
    config_path = sys.argv[1]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    interactive_mode = config.get("interactive", True)
    image_dir = config.get("image", {}).get("directory", None)
    if not image_dir or not os.path.isdir(image_dir) :
        print(f"Warning: Image directory '{image_dir}' not found or not specified in config. Image display will be skipped.")
        image_dir = None # Ensure it's None if invalid

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") # Unique ID for this run's files

    # Stage 1: Merge detection CSV files
    print("\n--- Stage 1: Merging Detection CSV Files ---")
    merged_detections_csv = config['csv']['merged_detections'].replace(".csv", f"_{run_id}.csv")
    merge_csv_files(
        config['csv']['detections1'],
        config['csv']['detections2'],
        merged_detections_csv
    )

    # Stage 2: Merge CSVs with bounding boxes
    print("\n--- Stage 2: Merging CSVs with Bounding Boxes ---")
    merged_bbox_csv = config['csv']['merged_bbox'].replace(".csv", f"_{run_id}.csv")
    merge_csvs_with_bbox(
        merged_detections_csv, # Use the output from Stage 1
        config['csv']['metadata'],
        merged_bbox_csv
    )

    # Stage 3: Update JSON annotations with timestamps
    print("\n--- Stage 3: Updating JSON Annotations with Timestamps ---")
    left_json_ts = config['json']['left']['output'].replace(".json", f"_{run_id}_ts.json")
    right_json_ts = config['json']['right']['output'].replace(".json", f"_{run_id}_ts.json")
    update_json_with_timestamp(config['json']['left']['input'], merged_bbox_csv, left_json_ts)
    update_json_with_timestamp(config['json']['right']['input'], merged_bbox_csv, right_json_ts)

    data_left = load_json_data(left_json_ts)
    data_right = load_json_data(right_json_ts)

    print_cluster_summary(data_left, "Initial Left")
    print_viewpoint_cluster_mapping(data_left, "left")
    print_cluster_summary(data_right, "Initial Right")
    print_viewpoint_cluster_mapping(data_right, "right")

    # Stage 4: Split Cluster Verification (Iterative & Interactive & Deterministic)
    print("\n--- Stage 4: Split Cluster Verification ---")
    grouped_left_split_stage = group_annotations_by_LCA_all(data_left)
    grouped_right_split_stage = group_annotations_by_LCA_all(data_right)

    # Iteratively apply consistency check for left view
    consistency_check_interactive_deterministic(grouped_left_split_stage, data_left, "Left", image_dir, interactive_mode, stage="split")
    # Update data_left annotations from potentially modified grouped_left_split_stage
    data_left['annotations'] = [ann for anns in grouped_left_split_stage.values() for ann in anns]
    
    # Iteratively apply consistency check for right view
    consistency_check_interactive_deterministic(grouped_right_split_stage, data_right, "Right", image_dir, interactive_mode, stage="split")
    data_right['annotations'] = [ann for anns in grouped_right_split_stage.values() for ann in anns]

    left_split_file = save_json_with_stage(data_left, config['json']['left']['output'], "split_verified", run_id)
    right_split_file = save_json_with_stage(data_right, config['json']['right']['output'], "split_verified", run_id)
    print_cluster_summary(data_left, "After Split Verification Left")
    print_cluster_summary(data_right, "After Split Verification Right")

    # Stage 6: Time-Overlap Verification (Interactive & Deterministic)
    # Note: grouped_... needs to be re-calculated as data_... might have changed (though LCA_ids not by split stage directly)
    grouped_left_time_stage = group_annotations_by_LCA_all(data_left) # Re-group
    grouped_right_time_stage = group_annotations_by_LCA_all(data_right) # Re-group

    print("\n--- Stage 6: Time-Overlap Verification ---")
    time_overlap_verification_interactive_deterministic(grouped_left_time_stage, data_left, "Left", image_dir, interactive_mode)
    data_left['annotations'] = [ann for anns in grouped_left_time_stage.values() for ann in anns]

    time_overlap_verification_interactive_deterministic(grouped_right_time_stage, data_right, "Right", image_dir, interactive_mode)
    data_right['annotations'] = [ann for anns in grouped_right_time_stage.values() for ann in anns]
    
    left_time_file = save_json_with_stage(data_left, left_split_file, "time_verified", "") # Use previous as base for name
    right_time_file = save_json_with_stage(data_right, right_split_file, "time_verified", "")
    print_cluster_summary(data_left, "After Time-Overlap Left")
    print_cluster_summary(data_right, "After Time-Overlap Right")

    # Stage 5: Cluster Equivalence & Final ID Assignment (Logic from Script 1)
    print("\n--- Stage 5: Cluster Equivalence and Final ID Assignment ---")
    # Group with viewpoint suffixes for equivalence checking
    grouped_left_wv_s5 = group_annotations_by_LCA_with_viewpoint(data_left, 'left')
    grouped_right_wv_s5 = group_annotations_by_LCA_with_viewpoint(data_right, 'right')
    
    print_viewpoint_cluster_mapping(data_left, "left") # Show current state
    print_viewpoint_cluster_mapping(data_right, "right")

    # Iterative conflict resolution and equivalence checking (from Script 1's main loop for Stage 5)
    max_s5_loops = 3 # Safeguard from Script 1's main
    final_eq_sets = []
    for s5_loop_idx in range(max_s5_loops):
        print(f"  [Stage 5] Equivalence & Splitting Cycle: {s5_loop_idx + 1}")
        current_eq_sets = check_cluster_equivalence_s1(grouped_left_wv_s5, grouped_right_wv_s5)
        
        # This function modifies grouped_X_wv and data_X['annotations'] in place
        made_splits_s5 = split_conflicting_clusters_iteratively_s1(grouped_left_wv_s5, grouped_right_wv_s5, data_left, data_right)
        
        if made_splits_s5:
            print("    Splits made in Stage 5. Re-grouping and re-checking equivalence...")
            # Crucial: Regenerate grouped_X_wv from modified data_X
            grouped_left_wv_s5 = group_annotations_by_LCA_with_viewpoint(data_left, 'left')
            grouped_right_wv_s5 = group_annotations_by_LCA_with_viewpoint(data_right, 'right')
            # Continue to next cycle of this Stage 5 loop
        else:
            print("    No splits made in this Stage 5 cycle. Conflict resolution stable.")
            final_eq_sets = current_eq_sets # Use the latest stable sets
            break 
    else: # If loop finished without break (max_s5_loops reached)
        print(f"    Reached max Stage 5 conflict resolution cycles ({max_s5_loops}). Using current state.")
        final_eq_sets = check_cluster_equivalence_s1(grouped_left_wv_s5, grouped_right_wv_s5) # Get final sets

    assign_ids_after_equivalence_check_s1(data_left, data_right, final_eq_sets)

    left_final_file = save_json_with_stage(data_left, left_time_file, "final_ids", "")
    right_final_file = save_json_with_stage(data_right, right_time_file, "final_ids", "")

    print_cluster_summary(data_left, "Final Left Viewpoint")
    print_cluster_summary(data_right, "Final Right Viewpoint")

    print("\nAll stages completed.")
    print(f"Final processed files (with run_id '{run_id}' in intermediates):")
    print(f"  Left: {left_final_file}")
    print(f"  Right: {right_final_file}")

if __name__ == "__main__":
    main()