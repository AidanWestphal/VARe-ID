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
import itertools
from PIL import Image

from VAREID.util.io.format_funcs import load_config, save_json, split_dataframe, join_dataframe_dict
from VAREID.util.utils import path_from_file

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
    from VAREID.util.ui.db_scripts import init_db, add_image_pair, get_decisions, check_pair_exists
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def wait_for_dropdown_value(dropdown):
    while dropdown.value == 'Select':
        if hasattr(sys, 'stdout'): plt.pause(0.1)
        else: time.sleep(0.1)
    return dropdown.value

def get_user_decision(prompt="Merge clusters? (Yes/No): ", interactive_mode=True):
    if interactive_mode and widgets is not None:
        dropdown = widgets.Dropdown(
            options=['Select', 'Yes', 'No'], value='Select', description=prompt,
            style={'description_width': 'initial'}, layout={'width': 'max-content'}
        )
        display(dropdown)
        decision = wait_for_dropdown_value(dropdown)
        dropdown.close()
        clear_output(wait=True)
        return decision
    else:
        while True:
            decision_input = input(prompt).strip().lower()
            if decision_input.startswith("y"): return "Yes"
            elif decision_input.startswith("n"): return "No"
            print("Invalid input. Please enter Yes or No.")

# NEW: Database Helper Functions (simplified for single decisions)
def wait_for_single_decision(db_path, pair_id, check_interval=5):
    """Wait for a specific pair to be completed"""
    print(f"Waiting for verification decision for pair {pair_id}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT cluster1, cluster2 FROM image_verification
        WHERE id = ? AND status IN ('awaiting', 'in_progress')
    """, (pair_id,))
    cluster1, cluster2 = cursor.fetchone()
    conn.close()

    print(f"Cluster 1: {cluster1} Cluster 2: {cluster2}")
    print("Please complete verification task in the UI...")
    
    while True:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT status FROM image_verification
            WHERE id = ? AND status IN ('awaiting', 'in_progress')
        """, (pair_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result is None:
            break
        
        print(f"Still waiting for cluster pair {cluster1} - {cluster2} - Checking again in {check_interval} seconds...")
        time.sleep(check_interval)
    
    print(f"Verification completed for cluster pair {cluster1} - {cluster2}!")

def get_single_decision(db_path, pair_id):
    """Get decision for a single pair"""
    decisions = get_decisions([pair_id], db_path)
    return decisions.get(pair_id)

def submit_pair_to_database(best_ann1, best_ann2, image_dir, db_path):
    """Submit a pair to database, returns pair_id and existing decision if any"""
    # Order UUIDs to create consistent ID
    uuid1 = best_ann1['uuid']
    uuid2 = best_ann2['uuid']
    if uuid1 > uuid2:
        uuid1, uuid2 = uuid2, uuid1
        best_ann1, best_ann2 = best_ann2, best_ann1

    unique_id = f"{uuid1}_{uuid2}"

    # Check if this pair already exists
    exists, status, decision = check_pair_exists(uuid1, uuid2, db_path)
    
    if exists:
        if status in ['checked', 'sent'] and decision != 'none':
            return {'pair_id': unique_id, 'decision': decision}
        else:
            return {'pair_id': unique_id, 'decision': None}

    # Get image paths from annotation's image_path field
    image_path1 = best_ann1.get('image_path')
    image_path2 = best_ann2.get('image_path')

    # Extract cluster information
    cluster1 = str(best_ann1.get('LCA_clustering_id', 'UNKNOWN'))
    cluster2 = str(best_ann2.get('LCA_clustering_id', 'UNKNOWN'))

    # Extract bounding boxes
    bbox1 = None
    bbox2 = None
    if "bbox" in best_ann1 and best_ann1["bbox"]:
        x, y, w, h = best_ann1["bbox"]
        bbox1 = json.dumps([x, y, x + w, y + h])
    if "bbox" in best_ann2 and best_ann2["bbox"]:
        x, y, w, h = best_ann2["bbox"]
        bbox2 = json.dumps([x, y, x + w, y + h])

    # Add to database
    add_image_pair(
        unique_id,
        uuid1,
        image_path1 or "NO_IMAGE",
        bbox1,
        cluster1,
        uuid2,
        image_path2 or "NO_IMAGE",
        bbox2,
        cluster2,
        db_path
    )
    
    return {'pair_id': unique_id, 'decision': None}

def save_json_with_stage(data, original_filename, stage_suffix, final=False):
    base, ext = os.path.splitext(original_filename)
    new_filename = f"{base}{ext}" if final else f"{base}_{stage_suffix}{ext}"
    df = pd.DataFrame(data["annotations"])
    final_data_to_save = split_dataframe(df)
    save_json(final_data_to_save, new_filename)
    print(f"Saved file: {new_filename}")
    return new_filename

def print_viewpoint_cluster_mapping(data, viewpoint):
    annotations = data.get("annotations", [])
    grouped = defaultdict(set)
    for ann in annotations:
        key = ann.get("LCA_clustering_id")
        if key is not None: grouped[key].add(ann.get("tracking_id"))
    print(f"\n--- {viewpoint.capitalize()} Viewpoint Cluster to Tracking ID Mapping ---")
    for cluster in sorted(grouped.keys()):
        print(f"  Cluster {cluster}: Tracking IDs: {grouped[cluster]}")

def get_cluster_best_ann_for_display(annotations_list):
    if not annotations_list: return None
    return max(annotations_list, key=lambda x: x.get("CA_score", 0.0))

def _update_cluster_merge_deterministic(grouped_annotations, source_cluster_id, target_cluster_id):
    if source_cluster_id not in grouped_annotations or target_cluster_id not in grouped_annotations: return
    if source_cluster_id == target_cluster_id: return
    print(f"    Merging cluster {source_cluster_id} into {target_cluster_id}...")
    base_target_id = grouped_annotations[target_cluster_id][0]['LCA_clustering_id']
    for ann in grouped_annotations[source_cluster_id]:
        ann['LCA_clustering_id'] = base_target_id
    grouped_annotations[target_cluster_id].extend(grouped_annotations[source_cluster_id])
    del grouped_annotations[source_cluster_id]

def _update_split_no_merge_deterministic(grouped_annotations, anchor_cluster_id, other_cluster_id):
    if anchor_cluster_id not in grouped_annotations or other_cluster_id not in grouped_annotations: return False
    anchor_tids = {ann['tracking_id'] for ann in grouped_annotations[anchor_cluster_id]}
    renamed_count = 0
    for ann in grouped_annotations[other_cluster_id]:
        if ann['tracking_id'] in anchor_tids:
            ann['tracking_id'] = f"{ann['tracking_id']}_new"
            renamed_count += 1
    if renamed_count:
        print(f"    No merge: Renamed {renamed_count} conflicting TIDs in cluster {other_cluster_id}.")
    return bool(renamed_count)

def group_annotations_by_lca(data):
    grouped = defaultdict(list)
    for ann in data.get('annotations', []):
        if (lca_id := ann.get('LCA_clustering_id')) is not None:
            grouped[lca_id].append(ann)
    return grouped

def pairwise_verification_interactive(grouped_annotations, c1_id, c2_id, image_dir, interactive_mode, stage, db_path=None):
    if (c1_id not in grouped_annotations or c2_id not in grouped_annotations or
            not grouped_annotations[c1_id] or not grouped_annotations[c2_id]):
        return False

    best_ann1 = get_cluster_best_ann_for_display(grouped_annotations[c1_id])
    best_ann2 = get_cluster_best_ann_for_display(grouped_annotations[c2_id])
    if not best_ann1 or not best_ann2: return False

    # Handle database mode internally
    if interactive_mode == "database":
        result = submit_pair_to_database(best_ann1, best_ann2, image_dir, db_path)
        
        if result['decision'] is not None:
            # Use existing decision
            decision = "Yes" if result['decision'] == 'correct' else "No"
        else:
            # Wait for new decision
            wait_for_single_decision(db_path, result['pair_id'])
            db_decision = get_single_decision(db_path, result['pair_id'])
            decision = "Yes" if db_decision == 'correct' else "No"
    else:
        # Original interactive/console mode
        print(f"\n[Stage: {stage.capitalize()}] User Verification: Clusters '{c1_id}' vs '{c2_id}'")
        # This check is now only a fallback; primary path is file_path from annotation
        if image_dir and os.path.isdir(image_dir):
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            for i, ann_ref in enumerate([best_ann1, best_ann2]):
                ax = axes[i]
                # RESTORED ORIGINAL, ROBUST METHOD: Use 'image_path' field directly.
                file_path = ann_ref.get('image_path')
                if file_path and os.path.exists(file_path):
                    try:
                        img = Image.open(file_path)
                        x, y, w, h = ann_ref['bbox']
                        ax.imshow(img.crop((x, y, x + w, y + h)))
                        ax.set_title(f"Cls:{ann_ref['LCA_clustering_id']}\n"
                                     f"TrkID:{ann_ref['tracking_id']}\n"
                                     f"UUID:{ann_ref.get('uuid','NA')}\n"
                                     f"CA:{ann_ref.get('CA_score',0):.3f}")
                        ax.axis('off')
                    except Exception as e:
                        ax.text(0.5, 0.5, f"Error: {e}", ha='center'); ax.axis('off')
                else:
                    ax.text(0.5, 0.5, "Image N/A", ha='center'); ax.axis('off')
            plt.tight_layout()
            if widgets and interactive_mode == True: display(fig)
            else: plt.show(block=False)

        decision = get_user_decision(f"    Merge {c1_id} & {c2_id}? (Yes/No): ", interactive_mode == True)

    # Apply decision (same logic for all modes)
    anchor_id, other_id = sorted([c1_id, c2_id])
    if decision == "Yes":
        print(f"    User chose to MERGE.")
        _update_cluster_merge_deterministic(grouped_annotations, other_id, anchor_id)
        return True
    else:
        if stage == 'split':
            return _update_split_no_merge_deterministic(grouped_annotations, anchor_id, other_id)
        print("    User chose NOT to merge.")
        return False

# -------------------------
# STAGE 1: TID SPLIT VERIFICATION
# -------------------------
def tid_split_verification_interactive(grouped_ann, data_view, viewpoint, image_dir, interactive, db_path=None):
    print(f"\n--- Verifying TID Splits for {viewpoint} Viewpoint ---")
    while True:
        tid_to_lca_map = defaultdict(set)
        for lca_id, anns in grouped_ann.items():
            for ann in anns: tid_to_lca_map[ann['tracking_id']].add(lca_id)
        
        conflicts = {tid: lcas for tid, lcas in tid_to_lca_map.items() if len(lcas) > 1}
        if not conflicts:
            print(f"  No TID splits found in {viewpoint}. Viewpoint is stable."); break
        
        changed_this_pass = False
        for tid, lca_set in sorted(conflicts.items(), key=lambda item: str(item[0])):
            print(f"  Conflict found: TID {tid} exists in clusters {lca_set}")
            for c1, c2 in itertools.combinations(sorted(lca_set), 2):
                if pairwise_verification_interactive(grouped_ann, c1, c2, image_dir, interactive, stage="split", db_path=db_path):
                    changed_this_pass = True; break
            if changed_this_pass: break
        
        if not changed_this_pass:
            print(f"  No more merges/renames needed for {viewpoint}. Viewpoint is stable."); break
        else:
            print(f"  Change occurred. Re-evaluating {viewpoint} for stability...")
            data_view['annotations'] = [ann for L in grouped_ann.values() for ann in L]

# -------------------------
# STAGE 2: TIME-OVERLAP VERIFICATION
# -------------------------
def parse_timestamp(ts_str):
    if not isinstance(ts_str, str): raise ValueError("Timestamp must be a string.")
    try: return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        try:
            ts_str = ts_str.replace(',', '.', 1)
            return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
        except:
            return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")

def intervals_overlap(start1, end1, start2, end2):
    return start1 <= end2 and start2 <= end1

def find_clusters_with_no_time_overlap(grouped_annotations):
    cluster_intervals = {}
    for lca_id, anns in grouped_annotations.items():
        timestamps = []
        for ann in anns:
            if 'timestamp' in ann and ann['timestamp']:
                try: timestamps.append(parse_timestamp(ann['timestamp']))
                except (ValueError, TypeError): continue
        if timestamps:
            cluster_intervals[lca_id] = (min(timestamps), max(timestamps))
    
    non_overlapping_pairs = []
    sorted_lca_ids = sorted(cluster_intervals.keys())
    for i in range(len(sorted_lca_ids)):
        for j in range(i + 1, len(sorted_lca_ids)):
            id1, id2 = sorted_lca_ids[i], sorted_lca_ids[j]
            if id1 not in cluster_intervals or id2 not in cluster_intervals: continue
            start1, end1 = cluster_intervals[id1]
            start2, end2 = cluster_intervals[id2]
            if not intervals_overlap(start1, end1, start2, end2):
                non_overlapping_pairs.append((id1, id2))
    return non_overlapping_pairs

def time_overlap_verification_interactive(grouped_ann, data_view, viewpoint, image_dir, interactive, db_path=None):
    print(f"\n--- Verifying Time Overlaps for {viewpoint} Viewpoint ---")
    while True:
        no_overlap_pairs = find_clusters_with_no_time_overlap(grouped_ann)
        if not no_overlap_pairs:
            print(f"  No more non-overlapping cluster pairs found in {viewpoint}. Stage complete."); break
        
        print(f"  Found {len(no_overlap_pairs)} candidate pairs with non-overlapping time intervals.")
        changed_this_pass = False
        for c1, c2 in no_overlap_pairs:
            if pairwise_verification_interactive(grouped_ann, c1, c2, image_dir, interactive, stage="time", db_path=db_path):
                changed_this_pass = True; break
        
        if not changed_this_pass:
            print(f"  No merges made in this pass for {viewpoint}. Time-overlap check is stable."); break
        else:
            print(f"  Merge occurred. Re-evaluating {viewpoint} for stability...")
            data_view['annotations'] = [ann for L in grouped_ann.values() for ann in L]

# -------------------------
# STAGE 3: UNLINKED CLUSTER VERIFICATION
# -------------------------
def get_parent_id(tid):
    return str(tid).split('_new')[0]

def unlinked_cluster_verification_interactive(grouped_ann, data_view, viewpoint, image_dir, interactive, db_path=None):
    """
    Interactively verifies and merges clusters within a single viewpoint that have
    no parent TID overlap. This version remembers 'No' decisions to avoid re-asking.
    """
    print(f"\n--- Verifying Unlinked Clusters for {viewpoint} Viewpoint ---")
    
    # This set will remember pairs the user has already declined in this stage.
    declined_pairs = set()

    while True:
        parent_id_map = {cid: {get_parent_id(ann['tracking_id']) for ann in anns} 
                         for cid, anns in grouped_ann.items()}
        
        cluster_ids = sorted(parent_id_map.keys())
        candidate_pairs = []
        for c1, c2 in itertools.combinations(cluster_ids, 2):
            # Exclude pairs that have already been declined
            if tuple(sorted((c1, c2))) in declined_pairs:
                continue
            
            if not parent_id_map.get(c1, set()).intersection(parent_id_map.get(c2, set())):
                candidate_pairs.append((c1, c2))
        
        if not candidate_pairs:
            print(f"  No new unlinked pairs to check in {viewpoint}. Stage complete."); break

        print(f"  Found {len(candidate_pairs)} new candidate pairs with no parent TID overlap.")
        changed_this_pass = False
        for c1, c2 in candidate_pairs:
            # A 'True' return value means a merge occurred.
            if pairwise_verification_interactive(grouped_ann, c1, c2, image_dir, interactive, stage="unlinked", db_path=db_path):
                changed_this_pass = True
                # A merge happened, so we must break and restart the main loop.
                # The declined_pairs set is preserved.
                break
            else:
                # No merge occurred, so the user selected "No". Remember this pair.
                declined_pairs.add(tuple(sorted((c1, c2))))
        
        if not changed_this_pass:
            print(f"  No merges made in this pass for {viewpoint}. Viewpoint is stable."); break
        else:
            print(f"  Merge occurred. Re-evaluating {viewpoint} for stability...")
            data_view['annotations'] = [ann for L in grouped_ann.values() for ann in L]

# -------------------------
# STAGE 5: CROSS-VIEW EQUIVALENCE
# -------------------------
def group_annotations_by_lca_with_viewpoint(data, viewpoint):
    grouped = defaultdict(list)
    for ann in data.get('annotations', []):
        if (lca_id := ann.get('LCA_clustering_id')) is not None:
            grouped[f"{lca_id}_{viewpoint}"].append(ann)
    return grouped

def check_numeric_equivalence(grouped_left_wv, grouped_right_wv):
    adj_list = {**{k: set() for k in grouped_left_wv}, **{k: set() for k in grouped_right_wv}}
    tid_to_clusters = defaultdict(set)
    all_grouped_wv = {**grouped_left_wv, **grouped_right_wv}
    for cluster_key, anns in all_grouped_wv.items():
        for ann in anns:
            tid = ann['tracking_id']
            if str(tid).isdigit():
                tid_to_clusters[tid].add(cluster_key)
    for clusters in tid_to_clusters.values():
        for c1, c2 in itertools.combinations(clusters, 2):
            if c1.endswith('_left') != c2.endswith('_left'):
                adj_list[c1].add(c2)
                adj_list[c2].add(c1)
    return adj_list

def find_conflicts(adj_list):
    conflicts_l, conflicts_r = defaultdict(set), defaultdict(set)
    for node, neighbors in adj_list.items():
        if len(neighbors) > 1:
            if node.endswith('_left'):
                conflicts_l[node] = neighbors
            else:
                conflicts_r[node] = neighbors
    return conflicts_l, conflicts_r

def generate_new_lca_id(base_id, existing_ids):
    i = 1
    while True:
        new_id = f"{base_id}_split{i}"
        if new_id not in existing_ids: return new_id
        i += 1

def split_conflicting_cluster(parent_key, targets, grouped_all_wv, all_lca_ids_in_view):
    print(f"    Conflict found: Cluster {parent_key} links to {len(targets)} other-view clusters.")
    parent_anns = list(grouped_all_wv[parent_key])
    parent_base_id = parent_key.split('_')[0]
    parent_view = parent_key.split('_')[1]
    
    newly_created_parts = defaultdict(list)
    moved_ann_uuids = set()

    for target_key in sorted(list(targets)):
        # --- THIS IS THE FIX ---
        # Check if the target cluster still exists before trying to access it.
        if target_key not in grouped_all_wv:
            print(f"      -> Skipping target {target_key} as it was already removed in this cycle.")
            continue
        # -----------------------

        target_numeric_tids = {ann['tracking_id'] for ann in grouped_all_wv[target_key] if str(ann['tracking_id']).isdigit()}
        anns_for_this_split = [ann for ann in parent_anns if ann['tracking_id'] in target_numeric_tids and ann['uuid'] not in moved_ann_uuids]
        
        if not anns_for_this_split: continue

        new_lca_id = generate_new_lca_id(parent_base_id, all_lca_ids_in_view)
        all_lca_ids_in_view.add(new_lca_id)
        new_cluster_key = f"{new_lca_id}_{parent_view}"
        print(f"      -> Creating new part {new_cluster_key} to link with {target_key}")

        for ann in anns_for_this_split:
            ann['LCA_clustering_id'] = new_lca_id
            newly_created_parts[new_cluster_key].append(ann)
            moved_ann_uuids.add(ann['uuid'])
    
    remaining_anns = [ann for ann in parent_anns if ann['uuid'] not in moved_ann_uuids]
    if not remaining_anns:
        del grouped_all_wv[parent_key]
    else:
        grouped_all_wv[parent_key] = remaining_anns
        print(f"      -> {len(remaining_anns)} annotations remain in {parent_key} as an isolated part.")

    for key, anns in newly_created_parts.items():
        grouped_all_wv[key] = anns
    return True

def handle_split_leftovers_interactive(grouped_wv, all_grouped_wv, image_dir, interactive, db_path=None):
    change_made_in_stage = False
    while True:
        base_id_to_siblings = defaultdict(list)
        for key in grouped_wv.keys():
            base_id_to_siblings[key.split('_split')[0]].append(key)

        made_merge_this_pass = False
        for base_key, siblings in base_id_to_siblings.items():
            if len(siblings) <= 1: continue

            adj_list = check_numeric_equivalence(all_grouped_wv, all_grouped_wv)
            isolated_siblings = {s for s in siblings if not adj_list.get(s)}
            linked_siblings = set(siblings) - isolated_siblings

            if not isolated_siblings or not linked_siblings: continue
            
            print(f"    Found isolated parts for base cluster {base_key.split('_')[0]}: {isolated_siblings}")
            for iso_c in sorted(list(isolated_siblings)):
                for lnk_c in sorted(list(linked_siblings)):
                    if iso_c not in all_grouped_wv or lnk_c not in all_grouped_wv: continue
                    if pairwise_verification_interactive(all_grouped_wv, iso_c, lnk_c, image_dir, interactive, stage="leftover_merge", db_path=db_path):
                        made_merge_this_pass = True
                        change_made_in_stage = True
                        break
                if made_merge_this_pass: break
            if made_merge_this_pass: break
        
        if not made_merge_this_pass:
            break
        else:
             # After a merge, grouped_wv is now stale, so we must rebuild it from the master group
            viewpoint = list(grouped_wv.keys())[0].split('_')[-1]
            current_view_keys = {k for k in all_grouped_wv if k.endswith(f'_{viewpoint}')}
            # Clear and repopulate grouped_wv
            grouped_wv.clear()
            for k in current_view_keys:
                grouped_wv[k] = all_grouped_wv[k]

    return change_made_in_stage

def assign_final_ids(grouped_left_wv, grouped_right_wv, data_left, data_right):
    print("\n--- Assigning Final IDs ---")
    adj_list = check_numeric_equivalence(grouped_left_wv, grouped_right_wv)
    
    visited, equivalence_sets, q = set(), [], []
    all_nodes = list(grouped_left_wv.keys()) + list(grouped_right_wv.keys())
    for node in all_nodes:
        if node not in visited:
            component, q = {node}, [node]
            visited.add(node)
            head = 0
            while head < len(q):
                curr = q[head]; head += 1
                for neighbor in adj_list.get(curr, []):
                    if neighbor not in visited:
                        visited.add(neighbor); component.add(neighbor); q.append(neighbor)
            equivalence_sets.append(component)

    final_id_map = {}
    print(f"  Found {len(equivalence_sets)} final equivalence sets (individuals).")
    for i, eq_set in enumerate(equivalence_sets):
        final_id = str(i + 1)
        print(f"  Individual {final_id}: Clusters {eq_set}")
        for cluster_key in eq_set: final_id_map[cluster_key] = final_id
    
    for ann in data_left['annotations']:
        ann['final_id'] = final_id_map.get(f"{ann['LCA_clustering_id']}_left", "UNASSIGNED")
    for ann in data_right['annotations']:
        ann['final_id'] = final_id_map.get(f"{ann['LCA_clustering_id']}_right", "UNASSIGNED")
    print("  Final ID assignment complete.")

# -------------------------
# MAIN WORKFLOW
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Post-process LCA outputs with interactive verification.")
    parser.add_argument("images", nargs='?', type=str, help="The image directory.")
    parser.add_argument("in_left", nargs='?', type=str, help="Path to the left LCA annotations JSON.")
    parser.add_argument("in_right", nargs='?', type=str, help="Path to the right LCA annotations JSON.")
    parser.add_argument("out_left", nargs='?', type=str, help="Path to save the processed left JSON.")
    parser.add_argument("out_right", nargs='?', type=str, help="Path to save the processed right JSON.")
    parser.add_argument("--db", type=str, help="Database path for verification (optional, can also be set in config as database.path)")
    parser.add_argument("--interaction_mode", type=str, choices=["console", "ipywidgets", "database"], help="Interaction mode (overrides config)")
    args = parser.parse_args()
    
    config = load_config(path_from_file(__file__, "postprocessing_config.yaml"))
    
    # NEW: Determine interaction mode - command line overrides config
    if args.interaction_mode:
        # Command line argument takes priority
        interaction_mode = args.interaction_mode
    elif "interaction_mode" in config:
        # New config format
        interaction_mode = config["interaction_mode"]
    else:
        # Backward compatibility with original config format
        interactive = config.get("interactive", True)
        if interactive == "database":
            interaction_mode = "database"
        elif interactive:
            interaction_mode = "ipywidgets"
        else:
            interaction_mode = "console"

    # Normalize interaction mode values for consistency
    if interaction_mode == "ipywidgets":
        interaction_mode = True
    elif interaction_mode == "console":
        interaction_mode = False
    # "database" stays as string

    # NEW: Use config values as defaults, override with command line args if provided
    images = args.images or config.get("image", {}).get("directory")
    in_left = args.in_left or config.get("json", {}).get("left", {}).get("input")
    in_right = args.in_right or config.get("json", {}).get("right", {}).get("input")
    out_left = args.out_left or config.get("json", {}).get("left", {}).get("output")
    out_right = args.out_right or config.get("json", {}).get("right", {}).get("output")

    # Validate required parameters
    if not in_left:
        raise ValueError("in_left is required. Specify as positional argument or set json.left.input in config.")
    if not in_right:
        raise ValueError("in_right is required. Specify as positional argument or set json.right.input in config.")
    if not out_left:
        raise ValueError("out_left is required. Specify as positional argument or set json.left.output in config.")
    if not out_right:
        raise ValueError("out_right is required. Specify as positional argument or set json.right.output in config.")

    # NEW: Validate database mode requirements and get database path
    db_path = None
    if interaction_mode == "database":
        if not DATABASE_AVAILABLE:
            raise ImportError("Database mode requires UI.db_scripts module. Please ensure database components are available.")
        
        # Get database path from config or command line
        db_path = args.db or config.get("database", {}).get("path")
        if db_path is None:
            raise ValueError("Database mode requires database path. Specify either --db argument or database.path in config file.")
        
        init_db(db_path)
        print(f"Using database mode with database: {db_path}")
    elif interaction_mode == True and widgets is None:
        print("Warning: ipywidgets not available, falling back to console mode")
        interaction_mode = False

    image_dir = images if images and os.path.exists(images) and os.path.isdir(images) else None
    if not image_dir:
        print(f"Warning: Image directory '{images}' not found or not specified. Image display will be skipped.")
    
    data_left = join_dataframe_dict(json.load(open(in_left)))
    data_right = join_dataframe_dict(json.load(open(in_right)))
    
    print("="*50 + "\nINITIAL DATA STATE\n" + "="*50)
    print_viewpoint_cluster_mapping(data_left, "left")
    print_viewpoint_cluster_mapping(data_right, "right")

    # --- STAGE 1 ---
    print("\n\n" + "="*50 + "\nSTAGE 1: TID SPLIT VERIFICATION\n" + "="*50)
    grouped_left = group_annotations_by_lca(data_left)
    grouped_right = group_annotations_by_lca(data_right)
    tid_split_verification_interactive(grouped_left, data_left, "Left", image_dir, interaction_mode, db_path)
    tid_split_verification_interactive(grouped_right, data_right, "Right", image_dir, interaction_mode, db_path)
    data_left['annotations'] = [ann for L in grouped_left.values() for ann in L]
    data_right['annotations'] = [ann for L in grouped_right.values() for ann in L]
    save_json_with_stage(data_left, out_left, "split_verified")
    save_json_with_stage(data_right, out_right, "split_verified")

    # --- STAGE 2 ---
    print("\n\n" + "="*50 + "\nSTAGE 2: TIME-OVERLAP VERIFICATION\n" + "="*50)
    time_overlap_verification_interactive(grouped_left, data_left, "Left", image_dir, interaction_mode, db_path)
    time_overlap_verification_interactive(grouped_right, data_right, "Right", image_dir, interaction_mode, db_path)
    data_left['annotations'] = [ann for L in grouped_left.values() for ann in L]
    data_right['annotations'] = [ann for L in grouped_right.values() for ann in L]
    save_json_with_stage(data_left, out_left, "time_verified")
    save_json_with_stage(data_right, out_right, "time_verified")

    # --- STAGE 3 ---
    print("\n\n" + "="*50 + "\nSTAGE 3: UNLINKED CLUSTER VERIFICATION\n" + "="*50)
    unlinked_cluster_verification_interactive(grouped_left, data_left, "Left", image_dir, interaction_mode, db_path)
    unlinked_cluster_verification_interactive(grouped_right, data_right, "Right", image_dir, interaction_mode, db_path)
    data_left['annotations'] = [ann for L in grouped_left.values() for ann in L]
    data_right['annotations'] = [ann for L in grouped_right.values() for ann in L]
    save_json_with_stage(data_left, out_left, "pre_equivalence")
    save_json_with_stage(data_right, out_right, "pre_equivalence")
    
    # --- STAGE 5 ---
    print("\n\n" + "="*50 + "\nSTAGE 5: CROSS-VIEW EQUIVALENCE & FINAL ID ASSIGNMENT\n" + "="*50)
    max_loops = 5
    for i in range(max_loops):
        print(f"\n--- Equivalence & Conflict Resolution Cycle {i+1}/{max_loops} ---")
        grouped_left_wv = group_annotations_by_lca_with_viewpoint(data_left, 'left')
        grouped_right_wv = group_annotations_by_lca_with_viewpoint(data_right, 'right')
        grouped_all_wv = {**grouped_left_wv, **grouped_right_wv}

        adj_list = check_numeric_equivalence(grouped_left_wv, grouped_right_wv)
        conflicts_l, conflicts_r = find_conflicts(adj_list)
        
        made_change_in_cycle = False
        if conflicts_l or conflicts_r:
            print("  Found one-to-many conflicts. Splitting clusters...")
            all_lca_ids_l = {key.split('_')[0] for key in grouped_left_wv}
            all_lca_ids_r = {key.split('_')[0] for key in grouped_right_wv}
            for conf_key, targets in conflicts_r.items():
                split_conflicting_cluster(conf_key, targets, grouped_all_wv, all_lca_ids_r)
            for conf_key, targets in conflicts_l.items():
                split_conflicting_cluster(conf_key, targets, grouped_all_wv, all_lca_ids_l)
            made_change_in_cycle = True
        else:
             print("  No one-to-many conflicts found based on numeric TIDs.")

        data_left['annotations'] = [ann for k, L in grouped_all_wv.items() if k.endswith('_left') for ann in L]
        data_right['annotations'] = [ann for k, L in grouped_all_wv.items() if k.endswith('_right') for ann in L]
        grouped_left_wv = group_annotations_by_lca_with_viewpoint(data_left, 'left')
        grouped_right_wv = group_annotations_by_lca_with_viewpoint(data_right, 'right')
        grouped_all_wv = {**grouped_left_wv, **grouped_right_wv}
        
        if handle_split_leftovers_interactive(grouped_left_wv, grouped_all_wv, image_dir, interaction_mode, db_path): made_change_in_cycle = True
        if handle_split_leftovers_interactive(grouped_right_wv, grouped_all_wv, image_dir, interaction_mode, db_path): made_change_in_cycle = True
        
        if not made_change_in_cycle:
            print("\n--- System is stable. No more splits or merges needed. ---")
            break
        else:
            data_left['annotations'] = [ann for k, L in grouped_all_wv.items() if k.endswith('_left') for ann in L]
            data_right['annotations'] = [ann for k, L in grouped_all_wv.items() if k.endswith('_right') for ann in L]
            print("  Changes were made in this cycle. Restarting...")
    else:
        print("\n--- Max reconciliation loops reached. Proceeding with current state. ---")

    final_grouped_left_wv = group_annotations_by_lca_with_viewpoint(data_left, 'left')
    final_grouped_right_wv = group_annotations_by_lca_with_viewpoint(data_right, 'right')
    assign_final_ids(final_grouped_left_wv, final_grouped_right_wv, data_left, data_right)

    # --- FINAL RESULTS ---
    print("\n\n" + "="*50 + "\nFINAL RESULTS\n" + "="*50)
    save_json_with_stage(data_left, out_left, "final", final=True)
    save_json_with_stage(data_right, out_right, "final", final=True)

    print_viewpoint_cluster_mapping(data_left, "left")
    print_viewpoint_cluster_mapping(data_right, "right")

    print("\n--- Final Individual ID Summary ---")
    final_summary = defaultdict(lambda: {'left': set(), 'right': set()})
    for ann in data_left['annotations']:
        final_summary[ann['final_id']]['left'].add(ann['LCA_clustering_id'])
    for ann in data_right['annotations']:
        final_summary[ann['final_id']]['right'].add(ann['LCA_clustering_id'])
    
    for fid in sorted(final_summary.keys(), key=lambda x: int(x) if x.isdigit() else float('inf')):
        print(f"Final ID {fid}:")
        print(f"  Left Clusters:  {sorted(list(final_summary[fid]['left'])) if final_summary[fid]['left'] else 'None'}")
        print(f"  Right Clusters: {sorted(list(final_summary[fid]['right'])) if final_summary[fid]['right'] else 'None'}")

    # NEW: Database mode info
    if interaction_mode == "database":
        print(f"\nDatabase verification completed using: {os.path.abspath(db_path)}")

if __name__ == "__main__":
    main()