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
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from PIL import Image

from util.format_funcs import save_json, split_dataframe, join_dataframe_dict

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

# Try to import database functions (NEW - for database mode)
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from UI.db_scripts import init_db, add_image_pair, get_decisions, check_pair_exists
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

# -------------------------
# Helper function for interactive waiting using ipywidgets.
# -------------------------
def wait_for_dropdown_value(dropdown):
    """Block until the dropdown value is changed from 'Select'."""
    while dropdown.value == 'Select':
        if hasattr(sys, 'stdout'):
             plt.pause(0.1)
        else:
            time.sleep(0.1)
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
            layout={'width': 'max-content'}
        )
        display(dropdown)
        decision = wait_for_dropdown_value(dropdown)
        dropdown.close()
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
# Database Helper Functions
# -------------------------
def wait_for_database_decisions(db_path, pair_ids, check_interval=5):
    """Wait for specific pairs to be completed"""
    if not pair_ids:
        return
    
    print(f"\nWaiting for {len(pair_ids)} verification decisions...")
    print("Please complete verification tasks in the UI...")
    
    while True:
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

def submit_pair_to_database(best_ann1, best_ann2, data_context, image_dir_path, db_path):
    """Submit a pair to database, returns pair_id or existing decision"""
    uuid1 = best_ann1['uuid']
    uuid2 = best_ann2['uuid']
    if uuid1 > uuid2:
        uuid1, uuid2 = uuid2, uuid1
        best_ann1, best_ann2 = best_ann2, best_ann1

    unique_id = f"{uuid1}_{uuid2}"

    exists, status, decision = check_pair_exists(uuid1, uuid2, db_path)
    
    if exists:
        if status in ['checked', 'sent'] and decision != 'none':
            return {'pair_id': unique_id, 'decision': decision}
        else:
            return {'pair_id': unique_id, 'decision': None}

    # Get image paths by looking them up in the data context
    image_path1 = None
    image_path2 = None
    for img in data_context.get("images", []):
        img_identifier = img.get("uuid") or img.get("image_uuid")
        if img_identifier:
            if img_identifier == best_ann1.get("image_uuid"):
                image_path1 = img.get("image_path")
            if img_identifier == best_ann2.get("image_uuid"):
                image_path2 = img.get("image_path")
        if image_path1 and image_path2:
            break
    
    bbox1 = None
    bbox2 = None
    if "bbox" in best_ann1 and best_ann1["bbox"]:
        x, y, w, h = best_ann1["bbox"]
        bbox1 = json.dumps([x, y, x + w, y + h])
    if "bbox" in best_ann2 and best_ann2["bbox"]:
        x, y, w, h = best_ann2["bbox"]
        bbox2 = json.dumps([x, y, x + w, y + h])

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
    
    return {'pair_id': unique_id, 'decision': None}
            
# -------------------------
# Helper: Save JSON file with stage suffix.
# -------------------------
def save_json_with_stage(data, original_filename, stage_suffix, run_identifier="", final=False):
    base, ext = os.path.splitext(original_filename)
    if final:
        new_filename = f"{base}{ext}"
    elif run_identifier:
        new_filename = f"{base}_{run_identifier}_{stage_suffix}{ext}"
    else:
        new_filename = f"{base}_{stage_suffix}{ext}"
    
    final_data_to_save = {
        "categories": data.get("categories", []),
        "images": data.get("images", []),
        "annotations": split_dataframe(pd.DataFrame(data.get("annotations", [])))
    }
    save_json(final_data_to_save, new_filename)
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
        for ann in grouped[cluster]:
            ca = ann.get("CA_score", 0)
            ind_id = ann.get("individual_id", "N/A")
            final_id = ann.get("final_id", "N/A")
            print(
                f"  TrkID: {ann.get('tracking_id')}, GT_IndID: {ind_id}, FinalID: {final_id}, "
                f"CA: {ca:.4f}, UUID: {ann.get('uuid')}"
            )
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


def group_annotations_by_LCA_all(data):
    annotations = data.get('annotations', [])
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
    if source_cluster_id not in grouped_annotations or target_cluster_id not in grouped_annotations: return
    if source_cluster_id == target_cluster_id: return

    grouped_annotations[target_cluster_id].extend(grouped_annotations[source_cluster_id])
    for ann in grouped_annotations[source_cluster_id]:
        ann['LCA_clustering_id'] = target_cluster_id
    del grouped_annotations[source_cluster_id]
    print(f"    Merged cluster {source_cluster_id} into {target_cluster_id}.")


def _update_split_no_merge_deterministic(grouped_annotations, anchor_cluster_id, other_cluster_id):
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
    data_context_for_display,
    image_dir_path, interactive_mode, stage, db_path=None
):
    if cluster1_id not in grouped_annotations or cluster2_id not in grouped_annotations or \
       not grouped_annotations[cluster1_id] or not grouped_annotations[cluster2_id]:
        print(f"  [{stage.capitalize()}] Skipping: {cluster1_id} or {cluster2_id} is missing/empty.")
        return False

    best_ann1 = get_cluster_best_ann_for_display(grouped_annotations[cluster1_id])
    best_ann2 = get_cluster_best_ann_for_display(grouped_annotations[cluster2_id])

    if not best_ann1 or not best_ann2:
        print(f"  [{stage.capitalize()}] Error: Could not get best annotations for display. Skipping.")
        return False

    if interactive_mode == "database":
        return submit_pair_to_database(best_ann1, best_ann2, data_context_for_display, image_dir_path, db_path)

    print(f"\n  [{stage.capitalize()} Stage] User Verification: Clusters '{cluster1_id}' and '{cluster2_id}'")
    
    # MODIFICATION: This block now looks up the full image path from the context.
    # It no longer relies on image_dir_path if the full path is available.
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for i, ann_ref in enumerate([best_ann1, best_ann2]):
        ax = axes[i]
        
        file_path = None
        for img in data_context_for_display.get("images", []):
            img_identifier = img.get("uuid") or img.get("image_uuid")
            if img_identifier and img_identifier == ann_ref.get("image_uuid"):
                file_path = img.get("image_path")
                break
        
        if file_path and os.path.exists(file_path):
            try:
                image = Image.open(file_path)
                x, y, w, h = ann_ref['bbox']
                cropped = image.crop((x, y, x + w, y + h))
                ax.imshow(cropped)
                ax.set_title(f"Cls: {ann_ref['LCA_clustering_id']}\nTrkID: {ann_ref['tracking_id']}\nUUID: {ann_ref.get('uuid', 'NA')}\nCA: {ann_ref.get('CA_score',0):.3f}")
                ax.axis('off')
            except Exception as e: ax.text(0.5,0.5,f"Err: {e}", ha='center'); ax.axis('off')
        else:
            ax.text(0.5,0.5, "Img N/F", ha='center'); ax.axis('off')
            print(f"Warning: Image not found at path: {file_path}")
            
    plt.tight_layout()
    if widgets and interactive_mode is True:
        display(fig)
    else:
        plt.show(block=False)

    decision = get_user_decision(prompt=f"    Merge {cluster1_id} & {cluster2_id}? (Yes/No): ", interactive_mode=(interactive_mode is True))
    if widgets and interactive_mode is True: clear_output(wait=True)

    anchor_id, other_id = sorted([cluster1_id, cluster2_id])

    if decision == "Yes":
        print(f"    User chose to MERGE. Anchor: {anchor_id}, Other: {other_id}")
        _update_cluster_merge_deterministic(grouped_annotations, other_id, anchor_id)
        return True
    else:
        print(f"    User chose NOT to merge. Anchor: {anchor_id}, Other: {other_id}")
        if stage == 'split':
            _update_split_no_merge_deterministic(grouped_annotations, anchor_id, other_id)
        return False

def consistency_check_interactive_deterministic(
    grouped_annotations_view, data_view,
    viewpoint_name, image_dir_path, interactive_mode, stage="split", db_path=None
):
    print(f"\n--- {viewpoint_name} Viewpoint {stage.capitalize()} Consistency Check ---")
    overall_changes_made = False
    while True:
        tid_to_lca_map = defaultdict(set)
        for lca_id, anns_list in grouped_annotations_view.items():
            for ann_item in anns_list:
                tid_to_lca_map[ann_item['tracking_id']].add(lca_id)
        
        made_merge_this_pass = False
        conflicting_tids = {tid: lcas for tid, lcas in tid_to_lca_map.items() if len(lcas) > 1}

        if not conflicting_tids:
            print(f"  No tracking ID splits found in {viewpoint_name} viewpoint.")
            break

        if interactive_mode == "database":
            pairs_to_decide = []
            pair_info = {}
            
            for tid, lca_set in conflicting_tids.items():
                print(f"  TID {tid} found in multiple LCAs: {lca_set} in {viewpoint_name}")
                sorted_lca_list = sorted(list(lca_set))
                c1_id, c2_id = sorted_lca_list[0], sorted_lca_list[1]
                
                result = pairwise_verification_interactive_deterministic(
                    grouped_annotations_view, c1_id, c2_id,
                    data_view, image_dir_path, interactive_mode, stage, db_path
                )
                
                if isinstance(result, dict):
                    if result['decision'] is not None:
                        anchor_id, other_id = sorted([c1_id, c2_id])
                        if result['decision'] == 'correct':
                            _update_cluster_merge_deterministic(grouped_annotations_view, other_id, anchor_id)
                            made_merge_this_pass = True
                            break
                        elif result['decision'] == 'incorrect' and stage == 'split':
                            _update_split_no_merge_deterministic(grouped_annotations_view, anchor_id, other_id)
                    else:
                        pairs_to_decide.append(result['pair_id'])
                        pair_info[result['pair_id']] = (c1_id, c2_id)
            
            if pairs_to_decide:
                wait_for_database_decisions(db_path, pairs_to_decide)
                decisions = get_decisions(pairs_to_decide, db_path)
                
                for pair_id in pairs_to_decide:
                    if pair_id in decisions and pair_id in pair_info:
                        decision = decisions[pair_id]
                        c1_id, c2_id = pair_info[pair_id]
                        anchor_id, other_id = sorted([c1_id, c2_id])
                        
                        if decision == 'correct':
                            if other_id in grouped_annotations_view and anchor_id in grouped_annotations_view:
                                _update_cluster_merge_deterministic(grouped_annotations_view, other_id, anchor_id)
                                made_merge_this_pass = True
                                break
                        elif decision == 'incorrect' and stage == 'split':
                            if anchor_id in grouped_annotations_view and other_id in grouped_annotations_view:
                                _update_split_no_merge_deterministic(grouped_annotations_view, anchor_id, other_id)
        else:
            for tid, lca_set in conflicting_tids.items():
                print(f"  TID {tid} found in multiple LCAs: {lca_set} in {viewpoint_name}")
                sorted_lca_list = sorted(list(lca_set))
                
                c1_id, c2_id = sorted_lca_list[0], sorted_lca_list[1]

                if pairwise_verification_interactive_deterministic(
                    grouped_annotations_view, c1_id, c2_id,
                    data_view, image_dir_path, interactive_mode, stage, db_path
                ):
                    overall_changes_made = True
                    made_merge_this_pass = True
                    break
        
        if not made_merge_this_pass:
            print(f"  No merges made in this pass for {viewpoint_name}. Consistency check for this viewpoint stable.")
            break
        else:
            print(f"  Merge occurred in {viewpoint_name}, re-evaluating consistency...")
            overall_changes_made = True
    return overall_changes_made


# =========================
# Stage 6: Time-Overlap Verification Helpers
# =========================
def parse_timestamp(ts_numeric):
    """MODIFICATION: Parses a numeric (unix-style) timestamp."""
    if not isinstance(ts_numeric, (int, float)):
        raise ValueError("Timestamp input must be a number.")
    return datetime.fromtimestamp(ts_numeric)

def intervals_overlap(start1, end1, start2, end2):
    return start1 <= end2 and start2 <= end1

def find_clusters_with_no_time_overlap(grouped_annotations, data_context):
    """MODIFICATION: Looks up numeric timestamps from the main images list."""
    
    uuid_to_timestamp = {}
    for img in data_context.get("images", []):
        img_identifier = img.get("uuid") or img.get("image_uuid")
        if img_identifier and 'timestamp' in img:
            uuid_to_timestamp[img_identifier] = img['timestamp']

    cluster_intervals = defaultdict(dict)
    
    for lca_id, anns_list in grouped_annotations.items():
        tid_to_times_map = defaultdict(list)
        for ann_item in anns_list:
            image_uuid = ann_item.get('image_uuid')
            if not image_uuid or image_uuid not in uuid_to_timestamp:
                continue
            
            timestamp_val = uuid_to_timestamp[image_uuid]
            try:
                datetime_obj = parse_timestamp(timestamp_val)
                tid_to_times_map[ann_item.get('tracking_id', 'UnknownTID')].append(datetime_obj)
            except (ValueError, TypeError):
                continue

        for tid, times_list in tid_to_times_map.items():
            if times_list:
                times_list.sort()
                cluster_intervals[lca_id][tid] = (times_list[0], times_list[-1])

    sorted_lca_ids = sorted(cluster_intervals.keys())
    non_overlapping_pairs = []

    for i in range(len(sorted_lca_ids)):
        for j in range(i + 1, len(sorted_lca_ids)):
            lca_id1 = sorted_lca_ids[i]
            lca_id2 = sorted_lca_ids[j]

            if not cluster_intervals[lca_id1] or not cluster_intervals[lca_id2]:
                continue

            overlap_found_between_lcas = False
            for tid1_intervals in cluster_intervals[lca_id1].values():
                start1, end1 = tid1_intervals
                for tid2_intervals in cluster_intervals[lca_id2].values():
                    start2, end2 = tid2_intervals
                    if intervals_overlap(start1, end1, start2, end2):
                        overlap_found_between_lcas = True
                        break
                if overlap_found_between_lcas:
                    break
            
            if not overlap_found_between_lcas:
                non_overlapping_pairs.append((lca_id1, lca_id2))
                
    return non_overlapping_pairs

# =========================
# Stage 6: Time-Overlap Verification Function (Interactive)
# =========================
def time_overlap_verification_interactive_deterministic(
    grouped_annotations_view, data_view, viewpoint_name,
    image_dir_path, interactive_mode, db_path=None
):
    print(f"\n--- {viewpoint_name} Viewpoint Time-Overlap Verification ---")
    no_overlap_pairs = find_clusters_with_no_time_overlap(grouped_annotations_view, data_view) 

    if not no_overlap_pairs:
        print(f"  All clusters in {viewpoint_name} have some time overlap or too few clusters to check.")
        return False

    print(f"  Found {len(no_overlap_pairs)} cluster pair(s) in {viewpoint_name} with NO time overlap.")
    
    if interactive_mode == "database":
        # ... (rest of function is unchanged)
        pairs_to_decide = []
        pair_info = {}
        any_merges = False
        
        for c1_id, c2_id in no_overlap_pairs:
            if c1_id not in grouped_annotations_view or c2_id not in grouped_annotations_view:
                continue
                
            print(f"  Verifying non-overlapping pair: {c1_id} & {c2_id}")
            result = pairwise_verification_interactive_deterministic(
                grouped_annotations_view, c1_id, c2_id,
                data_view, image_dir_path, interactive_mode, stage="time", db_path=db_path
            )
            
            if isinstance(result, dict):
                if result['decision'] is not None:
                    anchor_id, other_id = sorted([c1_id, c2_id])
                    if result['decision'] == 'correct':
                        for ann in grouped_annotations_view[other_id]:
                            ann["LCA_clustering_id"] = anchor_id
                        grouped_annotations_view[anchor_id].extend(grouped_annotations_view[other_id])
                        del grouped_annotations_view[other_id]
                        print(f"[Time Stage] Clusters {other_id} merged into {anchor_id}.")
                        any_merges = True
                    else:
                        print(f"[Time Stage] No merging performed for clusters {c1_id} and {c2_id}.")
                else:
                    pairs_to_decide.append(result['pair_id'])
                    pair_info[result['pair_id']] = (c1_id, c2_id)
        
        if pairs_to_decide:
            wait_for_database_decisions(db_path, pairs_to_decide)
            decisions = get_decisions(pairs_to_decide, db_path)
            
            for pair_id in pairs_to_decide:
                if pair_id in decisions and pair_id in pair_info:
                    decision = decisions[pair_id]
                    c1_id, c2_id = pair_info[pair_id]
                    
                    if decision == 'correct':
                        anchor_id, other_id = sorted([c1_id, c2_id])
                        if other_id in grouped_annotations_view and anchor_id in grouped_annotations_view:
                            for ann in grouped_annotations_view[other_id]:
                                ann["LCA_clustering_id"] = anchor_id
                            grouped_annotations_view[anchor_id].extend(grouped_annotations_view[other_id])
                            del grouped_annotations_view[other_id]
                            print(f"[Time Stage] Clusters {other_id} merged into {anchor_id}.")
                            any_merges = True
                    else:
                        print(f"[Time Stage] No merging performed for clusters {c1_id} and {c2_id}.")
        
        return any_merges
    else:
        any_merges_in_time_stage = False
        for c1_id, c2_id in no_overlap_pairs:
            if c1_id not in grouped_annotations_view or c2_id not in grouped_annotations_view:
                continue
            print(f"  Verifying non-overlapping pair: {c1_id} & {c2_id}")
            if pairwise_verification_interactive_deterministic(
                grouped_annotations_view, c1_id, c2_id,
                data_view, image_dir_path, interactive_mode, stage="time", db_path=db_path
            ):
                any_merges_in_time_stage = True
                print(f"    Merge occurred for {c1_id}, {c2_id}. Grouped annotations updated.")
        
        if any_merges_in_time_stage:
            print(f"  Merges were made in {viewpoint_name} during time-overlap verification.")
        else:
            print(f"  No merges made in {viewpoint_name} during time-overlap verification based on user decisions.")
        return any_merges_in_time_stage

def group_annotations_by_LCA_with_viewpoint(data, viewpoint):
    grouped = defaultdict(list)
    for ann in data.get('annotations', []):
        lca = ann.get('LCA_clustering_id')
        if lca is not None:
            grouped[f"{lca}_{viewpoint}"].append(ann)
    return grouped

# =========================
# Stage 5: Cluster Equivalence & Individual ID Assignment
# =========================
def check_cluster_equivalence_s1(grouped_left_wv, grouped_right_wv):
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
                if c1 in adj_list and c2 in adj_list :
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
                    if curr in adj_list:
                         component_stack.extend(adj_list[curr] - visited_nodes)
            if current_component:
                equivalence_sets.append(current_component)
    return equivalence_sets

def generate_new_lca_id_s1(base_lca_id_str, existing_lca_ids_in_viewpoint_data_set):
    base_lca_id_str = str(base_lca_id_str)
    max_suffix = 0
    pattern = re.compile(rf"^{re.escape(base_lca_id_str)}_split(\d+)$")
    for existing_id_str in existing_lca_ids_in_viewpoint_data_set:
        match = pattern.match(str(existing_id_str))
        if match: max_suffix = max(max_suffix, int(match.group(1)))
    new_suffix = max_suffix + 1
    return f"{base_lca_id_str}_split{new_suffix}"

def split_conflicting_clusters_iteratively_s1(grouped_left_wv, grouped_right_wv, data_left, data_right):
    overall_splits_made = False
    max_outer_loops = 5

    for outer_loop_count in range(max_outer_loops):
        split_made_in_this_pass = False

        all_current_right_lca_ids = {str(ann['LCA_clustering_id']) for ann in data_right['annotations'] if 'LCA_clustering_id' in ann}
        
        for r_cluster_key in list(grouped_right_wv.keys()): 
            if r_cluster_key not in grouped_right_wv or not grouped_right_wv[r_cluster_key]:
                if r_cluster_key in grouped_right_wv: del grouped_right_wv[r_cluster_key]
                continue

            original_r_anns_list = list(grouped_right_wv[r_cluster_key])
            r_cluster_tracking_ids = {ann['tracking_id'] for ann in original_r_anns_list}
            r_cluster_base_lca_id = str(original_r_anns_list[0]['LCA_clustering_id']) 

            tid_to_left_lca_map = defaultdict(set)
            mapped_left_lca_ids = set()
            for l_cluster_key_iter, l_anns_list_iter in grouped_left_wv.items():
                if not l_anns_list_iter: continue
                current_l_base_lca_id = str(l_anns_list_iter[0]['LCA_clustering_id'])
                for l_ann_iter in l_anns_list_iter:
                    if l_ann_iter['tracking_id'] in r_cluster_tracking_ids:
                        tid_to_left_lca_map[l_ann_iter['tracking_id']].add(current_l_base_lca_id)
                        mapped_left_lca_ids.add(current_l_base_lca_id)
            
            if len(mapped_left_lca_ids) > 1:
                print(f"    Conflict: Right Cluster {r_cluster_key} (Base LCA: {r_cluster_base_lca_id}) maps to multiple Left LCAs: {mapped_left_lca_ids}")
                split_made_in_this_pass = True
                overall_splits_made = True
                
                newly_created_split_parts = defaultdict(list)
                moved_ann_uuids_this_conflict = set()

                for target_left_lca_id_str in sorted(list(mapped_left_lca_ids)):
                    tids_for_this_specific_connection = {
                        tid for tid, mapped_lcas in tid_to_left_lca_map.items() if target_left_lca_id_str in mapped_lcas
                    }
                    if not tids_for_this_specific_connection: continue

                    new_r_lca_id_for_this_split = generate_new_lca_id_s1(r_cluster_base_lca_id, all_current_right_lca_ids)
                    all_current_right_lca_ids.add(new_r_lca_id_for_this_split)
                    new_r_cluster_key_for_this_split = f"{new_r_lca_id_for_this_split}_right"
                    print(f"      Defining new Right split part: {new_r_cluster_key_for_this_split} (for TIDs linking to Left LCA {target_left_lca_id_str})")

                    for r_ann in original_r_anns_list:
                        if r_ann['tracking_id'] in tids_for_this_specific_connection and \
                           r_ann['uuid'] not in moved_ann_uuids_this_conflict:
                            
                            ann_copy_for_split = r_ann.copy()
                            ann_copy_for_split['LCA_clustering_id'] = new_r_lca_id_for_this_split
                            newly_created_split_parts[new_r_cluster_key_for_this_split].append(ann_copy_for_split)
                            moved_ann_uuids_this_conflict.add(r_ann['uuid'])
                
                remaining_r_anns = [ann for ann in original_r_anns_list if ann['uuid'] not in moved_ann_uuids_this_conflict]
                if not remaining_r_anns:
                    if r_cluster_key in grouped_right_wv: del grouped_right_wv[r_cluster_key]
                else:
                    grouped_right_wv[r_cluster_key] = remaining_r_anns
                
                for new_key, new_anns in newly_created_split_parts.items():
                    if new_anns: grouped_right_wv[new_key] = new_anns

                updated_full_data_right_annotations = []
                for anns_in_cluster in grouped_right_wv.values(): 
                    updated_full_data_right_annotations.extend(anns_in_cluster)
                data_right['annotations'] = updated_full_data_right_annotations
                break 
        
        if split_made_in_this_pass: continue

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
                break
        
        if not split_made_in_this_pass:
            print(f"  [Conflict Resolution by Splitting] Stable after {outer_loop_count + 1} outer loop(s).")
            return overall_splits_made 

    print(f"  [Conflict Resolution by Splitting] Reached max outer loops ({max_outer_loops}).")
    return overall_splits_made

def assign_ids_after_equivalence_check_s1(data_left, data_right, eq_sets_from_s1_check):
    print("\n[Stage 5] Assigning Final IDs (Script 1 logic)")
    assigned_final_ids_map = {}
    next_available_numeric_id = 1

    for eq_group in eq_sets_from_s1_check:
        viewpoints_in_group = {cl_key.rsplit("_", 1)[1] for cl_key in eq_group}
        if "left" in viewpoints_in_group and "right" in viewpoints_in_group:
            base_lca_ids_in_group = {cl_key.rsplit('_',1)[0] for cl_key in eq_group}
            print(f"  Equivalent Group {base_lca_ids_in_group} (spans L/R). Assigning Final ID: {next_available_numeric_id}")
            for cl_key in eq_group:
                assigned_final_ids_map[cl_key] = str(next_available_numeric_id)
            next_available_numeric_id += 1
    
    all_left_cluster_keys = {f"{ann['LCA_clustering_id']}_left" for ann in data_left['annotations'] if 'LCA_clustering_id' in ann}
    for l_cl_key in sorted(list(all_left_cluster_keys)):
        if l_cl_key not in assigned_final_ids_map:
            base_lca = l_cl_key.rsplit("_", 1)[0]
            print(f"  Unmatched Left Cluster {base_lca}. Assigning New Final ID: {next_available_numeric_id}")
            assigned_final_ids_map[l_cl_key] = str(next_available_numeric_id)
            next_available_numeric_id +=1

    all_right_cluster_keys = {f"{ann['LCA_clustering_id']}_right" for ann in data_right['annotations'] if 'LCA_clustering_id' in ann}
    for r_cl_key in sorted(list(all_right_cluster_keys)):
        if r_cl_key not in assigned_final_ids_map:
            base_lca = r_cl_key.rsplit("_", 1)[0]
            temp_id = f"temp_{base_lca}_R"
            print(f"  Unmatched Right Cluster {base_lca}. Assigning Temporary Final ID: {temp_id}")
            assigned_final_ids_map[r_cl_key] = temp_id

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
# Utility JSON I/O Functions
# =========================
def load_json_data(file_path):
    with open(file_path, 'r') as f: return json.load(f)

# =========================
# Main Workflow
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Post process LCA outputs"
    )
    parser.add_argument(
        "images", nargs='?', type=str, help="The image directory."
    )
    parser.add_argument(
        "in_left", nargs='?', type=str, help="The full path to the left annotations for LCA.",
    )
    parser.add_argument(
        "in_right", nargs='?', type=str, help="The full path to the right annotations for LCA.",
    )
    parser.add_argument(
        "out_left", nargs='?', type=str, help="The path to save the processed left LCA json."
    )
    parser.add_argument(
        "out_right", nargs='?', type=str, help="The path to save the processed right LCA json."
    )
    parser.add_argument(
        "--db", type=str, help="Database path for verification"
    )
    parser.add_argument(
        "--config", type=str, help="Path to the config file"
    )

    args = parser.parse_args()

    # MODIFICATION: Restored original config handling logic
    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    db_path = None
    if "interaction_mode" in config:
        interaction_mode = config["interaction_mode"]
    else:
        interactive = config.get("interactive", True)
        interaction_mode = interactive

    if interaction_mode == "database":
        db_path = args.db or config.get("database", {}).get("path")
        if db_path and DATABASE_AVAILABLE:
            init_db(db_path)

    image_dir = args.images or config.get("image", {}).get("directory")
    
    in_left = args.in_left or config.get("json", {}).get("left", {}).get("input")
    in_right = args.in_right or config.get("json", {}).get("right", {}).get("input")
    out_left = args.out_left or config.get("json", {}).get("left", {}).get("output")
    out_right = args.out_right or config.get("json", {}).get("right", {}).get("output")

    if not in_left or not in_right or not out_left or not out_right:
        raise ValueError("Input and output JSON paths are required.")

    # Using the user-provided utility function to load data
    data_left = join_dataframe_dict(load_json_data(in_left))
    data_right = join_dataframe_dict(load_json_data(in_right))

    # --- The rest of main is unchanged ---
    print_cluster_summary(data_left, "Initial Left")
    print_viewpoint_cluster_mapping(data_left, "left")
    print_cluster_summary(data_right, "Initial Right")
    print_viewpoint_cluster_mapping(data_right, "right")

    print("\n--- Stage 4: Split Cluster Verification ---")
    grouped_left_split_stage = group_annotations_by_LCA_all(data_left)
    grouped_right_split_stage = group_annotations_by_LCA_all(data_right)

    consistency_check_interactive_deterministic(grouped_left_split_stage, data_left, "Left", image_dir, interaction_mode, stage="split", db_path=db_path)
    data_left['annotations'] = [ann for anns in grouped_left_split_stage.values() for ann in anns]
    
    consistency_check_interactive_deterministic(grouped_right_split_stage, data_right, "Right", image_dir, interaction_mode, stage="split", db_path=db_path)
    data_right['annotations'] = [ann for anns in grouped_right_split_stage.values() for ann in anns]

    left_split_file = save_json_with_stage(data_left, out_left, "split_verified")
    right_split_file = save_json_with_stage(data_right, out_right, "split_verified")
    print_cluster_summary(data_left, "After Split Verification Left")
    print_cluster_summary(data_right, "After Split Verification Right")

    grouped_left_time_stage = group_annotations_by_LCA_all(data_left)
    grouped_right_time_stage = group_annotations_by_LCA_all(data_right)

    print("\n--- Stage 6: Time-Overlap Verification ---")
    time_overlap_verification_interactive_deterministic(grouped_left_time_stage, data_left, "Left", image_dir, interaction_mode, db_path)
    data_left['annotations'] = [ann for anns in grouped_left_time_stage.values() for ann in anns]

    time_overlap_verification_interactive_deterministic(grouped_right_time_stage, data_right, "Right", image_dir, interaction_mode, db_path)
    data_right['annotations'] = [ann for anns in grouped_right_time_stage.values() for ann in anns]
    
    left_time_file = save_json_with_stage(data_left, left_split_file, "time_verified", "")
    right_time_file = save_json_with_stage(data_right, right_split_file, "time_verified", "")
    print_cluster_summary(data_left, "After Time-Overlap Left")
    print_cluster_summary(data_right, "After Time-Overlap Right")

    print("\n--- Stage 5: Cluster Equivalence and Final ID Assignment ---")
    grouped_left_wv_s5 = group_annotations_by_LCA_with_viewpoint(data_left, 'left')
    grouped_right_wv_s5 = group_annotations_by_LCA_with_viewpoint(data_right, 'right')
    
    print_viewpoint_cluster_mapping(data_left, "left")
    print_viewpoint_cluster_mapping(data_right, "right")

    max_s5_loops = 3
    final_eq_sets = []
    for s5_loop_idx in range(max_s5_loops):
        print(f"  [Stage 5] Equivalence & Splitting Cycle: {s5_loop_idx + 1}")
        current_eq_sets = check_cluster_equivalence_s1(grouped_left_wv_s5, grouped_right_wv_s5)
        
        made_splits_s5 = split_conflicting_clusters_iteratively_s1(grouped_left_wv_s5, grouped_right_wv_s5, data_left, data_right)
        
        if made_splits_s5:
            print("    Splits made in Stage 5. Re-grouping and re-checking equivalence...")
            grouped_left_wv_s5 = group_annotations_by_LCA_with_viewpoint(data_left, 'left')
            grouped_right_wv_s5 = group_annotations_by_LCA_with_viewpoint(data_right, 'right')
        else:
            print("    No splits made in this Stage 5 cycle. Conflict resolution stable.")
            final_eq_sets = current_eq_sets
            break 
    else:
        print(f"    Reached max Stage 5 conflict resolution cycles ({max_s5_loops}). Using current state.")
        final_eq_sets = check_cluster_equivalence_s1(grouped_left_wv_s5, grouped_right_wv_s5)

    assign_ids_after_equivalence_check_s1(data_left, data_right, final_eq_sets)

    left_final_file = save_json_with_stage(data_left, left_time_file, "final_ids", final=True)
    right_final_file = save_json_with_stage(data_right, right_time_file, "final_ids", final=True)

    print_cluster_summary(data_left, "Final Left Viewpoint")
    print_cluster_summary(data_right, "Final Right Viewpoint")

    print("\nAll stages completed.")
    print(f"Final processed files:")
    print(f"  Left: {left_final_file}")
    print(f"  Right: {right_final_file}")

    if interaction_mode == "database" and db_path:
        print(f"\nDatabase verification completed using: {os.path.abspath(db_path)}")

if __name__ == "__main__":
    main()