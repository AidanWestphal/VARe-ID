# import json
# import yaml
# import random
# import argparse
# from collections import defaultdict
# from copy import deepcopy
# import pandas as pd
# import shutil

# from VAREID.libraries.io.format_funcs import load_config, load_json, save_json, split_dataframe, join_dataframe_dict
# from VAREID.libraries.utils import path_from_file

# def group_annotations_by_tracking_id_and_subsequences(data):
#     annotations = data['annotations']
#     by_tid = defaultdict(list)

#     for ann in annotations:
#         by_tid[ann['tracking_id']].append(ann)

#     subseqs = {}
#     for tid, lst in by_tid.items():
#         lst.sort(key=lambda x: x['frame_number'])
#         cur, prev = [], None
#         out = []
#         for ann in lst:
#             if prev is None or ann['frame_number'] == prev + 1:
#                 cur.append(ann)
#             else:
#                 out.append(cur)
#                 cur = [ann]
#             prev = ann['frame_number']
#         if cur:
#             out.append(cur)
#         subseqs[tid] = out

#     return subseqs

# def frame_sampling_algorithm_combined(data, t_seconds, frame_interval, threshold_pct, ca_available, viewpoint_available):
#     data = deepcopy(data)
#     print(f"\nInitial annotations: {len(data['annotations'])}")

#     # Bucket by viewpoint if enabled
#     buckets = defaultdict(list)
#     for ann in data['annotations']:
#         vp = ann.get('viewpoint', 'unknown') if viewpoint_available else 'unknown'
#         buckets[vp].append(ann)

#     t_frames = max(1, int(round(t_seconds * frame_interval)))
#     out = []

#     for vp, anns in buckets.items():
#         print(f"\nViewpoint = {vp}, #anns = {len(anns)}")
#         subseqs = group_annotations_by_tracking_id_and_subsequences({
#             'images': data['images'],
#             'annotations': anns
#         })
#         for tid, seqs in subseqs.items():
#             for i, seq in enumerate(seqs, 1):
#                 print(f"  TID {tid}, subseq #{i}, len={len(seq)}")
#                 if ca_available:
#                     best = max(seq, key=lambda x: x['CA_score'])
#                     max_ca = best['CA_score']
#                     thr = max_ca * threshold_pct
#                     print(f"    max CA={max_ca:.3f}, thr={thr:.3f}")
#                     selected = [best]
#                     frames = {best['frame_number']}

#                     seq_sorted = sorted(seq, key=lambda x: x['frame_number'])
#                     for a_prev, a_cur, a_next in zip(seq_sorted, seq_sorted[1:], seq_sorted[2:]):
#                         if (a_cur['CA_score'] >= a_prev['CA_score']
#                             and a_cur['CA_score'] >= a_next['CA_score']
#                             and a_cur['CA_score'] >= thr):
#                             fn = a_cur['frame_number']
#                             if all(abs(fn - f) >= t_frames for f in frames):
#                                 selected.append(a_cur)
#                                 frames.add(fn)
#                                 print(f"      + local max @ frame {fn}")
#                 else:
#                     best = random.choice(seq)
#                     print(f"    (no CA) randomly picked frame {best['frame_number']}")
#                     selected = [best]

#                 out.extend(selected)

#     print(f"\nAfter Stage 1: {len(out)} annotations")
#     data['annotations'] = out
#     return data

# def group_annotations_by_tracking_id_and_viewpoint(data, viewpoint_available):
#     groups = {}
#     for ann in data['annotations']:
#         vp = ann.get('viewpoint', 'unknown') if viewpoint_available else 'unknown'
#         if vp not in groups:
#             groups[vp] = defaultdict(list)
#         groups[vp][ann['tracking_id']].append(ann)
#     return groups

# def select_highest_score_per_track(data, ca_available, viewpoint_available):
#     """
#     Selects only the single best annotation for each tracking ID per viewpoint.
#     - If ca_available, 'best' is the one with the highest CA_score.
#     - Otherwise, 'best' is chosen randomly.
#     """
#     data = deepcopy(data)
#     print(f"Initial annotations: {len(data['annotations'])}")
    
#     grouped_annotations = group_annotations_by_tracking_id_and_viewpoint(data, viewpoint_available)
#     final_annotations = []

#     for vp, by_tid in grouped_annotations.items():
#         for tid, anns in by_tid.items():
#             if not anns:
#                 continue
            
#             if ca_available:
#                 best_ann = max(anns, key=lambda x: x.get('CA_score', 0))
#                 score = best_ann.get('CA_score', 0)
#                 print(f"  VP={vp}, TID={tid}: Selected annotation with highest CA score ({score:.3f}) from {len(anns)} candidates.")
#             else:
#                 best_ann = random.choice(anns)
#                 print(f"  VP={vp}, TID={tid}: Randomly selected 1 annotation from {len(anns)} candidates (CA score not used).")
            
#             final_annotations.append(best_ann)
    
#     data['annotations'] = final_annotations
#     return data

# def filter_annotations(annotations_by_viewpoint, threshold_pct, ca_available):
#     for vp, by_tid in annotations_by_viewpoint.items():
#         for tid, anns in by_tid.items():
#             if ca_available and threshold_pct > 0:
#                 high = max(a['CA_score'] for a in anns)
#                 thr = high * threshold_pct
#                 filtered = [a for a in anns if a['CA_score'] >= thr]
#                 print(f"\nVP={vp} TID={tid}: {len(anns)}→{len(filtered)} by CA thr")
#             else:
#                 if threshold_pct > 0:
#                     num_keep = max(1, int(round(len(anns) * threshold_pct)))
#                 else:
#                     num_keep = len(anns)
#                 filtered = random.sample(anns, num_keep) if not ca_available else anns
#                 print(f"\nVP={vp} TID={tid}: {len(anns)}→{len(filtered)} random keep")
#             by_tid[tid] = filtered
#     return annotations_by_viewpoint

# def reconstruct_annotations(data, annotations_by_viewpoint):
#     new = []
#     for by_tid in annotations_by_viewpoint.values():
#         for anns in by_tid.values():
#             new.extend(anns)
#     data['annotations'] = new
#     return data

# def main():

#     # input_file: "ca_filtered_merged_tracking_ids_dji_0087_0088_no_unwanted_viewpoints_pre_miewid.json"
#     # final_output: "out2.json"
#     # step1_output: "out1.json"
#     # "python {input.script} {input.file} {output.json_stage1} {output.json_final}"

#     parser = argparse.ArgumentParser(
#         description="Run frame sampling algorithm on ia classifier output"
#     )
#     parser.add_argument(
#         "in_json_path",
#         type=str,
#         help="The full path to the ia classifier output json to use as input",
#     )
#     parser.add_argument(
#         "json_final", type=str, help="The full path to the output json file for final"
#     )
#     parser.add_argument(
#         "--json_stage1",
#         type=str, 
#         default=None,
#         help="The full path to the output json file for stage 1 (only used in 'many' mode)",
#     )
#     args = parser.parse_args()

#     # Load configuration from YAML file
#     cfg = load_config(path_from_file(__file__, "frame_sampling_config.yaml"))
#     in_file = args.in_json_path
#     step1_file = args.json_stage1
#     final_out = args.json_final

#     # Get common settings
#     ca_flag = cfg['settings'].get('use_ca_score', True)
#     vc_flag = cfg['settings'].get('use_viewpoint', True)
#     selection_mode = cfg['settings'].get('selection_mode', 'many') # Default to 'many'

#     random.seed(123456789)
#     data = load_json(in_file)
#     data = join_dataframe_dict(data)

#     if selection_mode == 'one':
#         print("\n=== Running the Single Annotation Selection Mode ===\n")
#         final_data = select_highest_score_per_track(data, ca_flag, vc_flag)
#         print(f"\nFinal annotations: {len(final_data['annotations'])}")
#         final_data = split_dataframe(pd.DataFrame(final_data['annotations']))
#         save_json(final_data, final_out)
#         print(f"Saved final output: {final_out}")

#     elif selection_mode == 'many':
#         print("\n=== Running the Many-Frame Sampling Mode ===\n")
#         t_sec = cfg['thresholds']['t_seconds']
#         frame_int = cfg['thresholds']['frame_interval']
#         pct1 = cfg['thresholds']['threshold_percentage_stage1']
#         pct2 = cfg['thresholds']['threshold_percentage_stage2']

#         print("=== Stage 1 ===")
#         stage1 = frame_sampling_algorithm_combined(
#             data, t_sec, frame_int, pct1,
#             ca_available=ca_flag,
#             viewpoint_available=vc_flag
#         )
        
#         if step1_file:
#             save_json(stage1, step1_file)
#             print(f"Saved Stage1 output: {step1_file}")

#         print("\n=== Stage 2 ===")
#         d2 = stage1
#         print(f"Loading {len(d2['annotations'])} annotations for Stage 2")
#         grouped = group_annotations_by_tracking_id_and_viewpoint(d2, vc_flag)
#         filtered = filter_annotations(grouped, pct2, ca_flag)
#         final_data = reconstruct_annotations(d2, filtered)
#         print(f"Final annotations: {len(final_data['annotations'])}")
#         final_data = split_dataframe(pd.DataFrame(final_data['annotations']))
#         save_json(final_data, final_out)
#         print(f"Saved final output → {final_out}")
#     else:
#         raise ValueError(f"Unknown selection_mode: '{selection_mode}'. Please use 'one' or 'many'.")

# if __name__ == "__main__":
#     main()




import json
import random
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from VAREID.libraries.io.format_funcs import load_config, load_json, save_json, split_dataframe, join_dataframe_dict
from VAREID.libraries.utils import path_from_file

# ============================================================================
# STRATIFICATION LOGIC
# ============================================================================

def get_stratified_ras(anns, num_ras):
    """
    Final step: Divide candidates into temporal bins and pick the highest
    CA_score frame from each bin. Ties favor the earlier frame.
    """
    if len(anns) <= num_ras:
        return anns
    
    anns.sort(key=lambda x: x['frame_number'])
    n = len(anns)
    bins = np.array_split(range(n), num_ras)
    
    selected = []
    for bin_indices in bins:
        if len(bin_indices) == 0: continue
        segment = [anns[i] for i in bin_indices]
        # Tie-breaker: Highest CA score first; if equal, earliest frame first
        # Sorting (CA, frame) in reverse order means higher CA wins, then lower frame wins
        segment.sort(key=lambda a: (a.get('CA_score', 0), -a['frame_number']), reverse=True)
        selected.append(segment[0])
    
    return selected

# ============================================================================
# 'MANY' MODE: ROBUST TEMPORAL NMS
# ============================================================================

def run_many_mode_logic(data, cfg):
    t_sec = cfg['thresholds']['t_seconds']
    f_int = cfg['thresholds']['frame_interval']
    t_frames = max(1, int(round(t_sec * f_int)))
    qual_pct = cfg['thresholds'].get('quality_threshold_pct', 0.8)
    num_ras = cfg['settings'].get('num_RA_annots', 5)
    use_vp = cfg['settings'].get('use_viewpoint', True)
    use_ca_master = cfg['settings'].get('use_ca_score', True)

    processed_annots = []
    by_vp_tid = defaultdict(lambda: defaultdict(list))
    
    for ann in data['annotations']:
        vp = ann.get('viewpoint', 'unknown') if use_vp else 'unknown'
        by_vp_tid[vp][ann['tracking_id']].append(ann)

    for vp, tracks in by_vp_tid.items():
        for tid, anns in tracks.items():
            anns.sort(key=lambda x: x['frame_number'])
            
            # 1. Quality baseline per track
            use_ca_for_track = use_ca_master
            if use_ca_master:
                ca_vals = [a.get('CA_score') for a in anns if a.get('CA_score') is not None]
                if not ca_vals:
                    use_ca_for_track = False
                else:
                    global_max = max(ca_vals)
                    qual_thresh = global_max * qual_pct

            if not use_ca_for_track:
                global_max, qual_thresh = 1.0, -float('inf')

            # 2. Candidate Selection (Interior Peaks)
            candidates = []
            for i, cur in enumerate(anns):
                score = cur.get('CA_score', 1.0) if use_ca_for_track else 1.0
                if score < qual_thresh: continue
                
                # Check neighbors for interior peak status
                if 0 < i < len(anns) - 1:
                    prev_s = anns[i-1].get('CA_score', -1e9) if use_ca_for_track else 1.0
                    next_s = anns[i+1].get('CA_score', -1e9) if use_ca_for_track else 1.0
                    if score >= prev_s and score >= next_s:
                        candidates.append(cur)
                elif len(anns) <= 2:
                    candidates.append(cur)

            if not candidates: continue

            # 3. TRUE NMS-IN-TIME (Deterministic Greedy Selection)
            # Higher CA score preferred; earlier frame number preferred for ties
            candidates.sort(key=lambda a: (a.get('CA_score', 0), -a['frame_number']), reverse=True)

            selected_nms, selected_frames = [], []
            for cand in candidates:
                fn = cand['frame_number']
                if all(abs(fn - sfn) >= t_frames for sfn in selected_frames):
                    selected_nms.append(cand)
                    selected_frames.append(fn)

            # 4. Final Stratification (Temporal Coverage)
            spaced_candidates = sorted(selected_nms, key=lambda a: a['frame_number'])
            final_ras = get_stratified_ras(spaced_candidates, num_ras)
            processed_annots.extend(final_ras)

    return processed_annots

# ============================================================================
# MAIN
# ============================================================================

# def main():
#     parser = argparse.ArgumentParser(description="KABR Unified Sampling - Final")
#     parser.add_argument("in_json_path", type=str)
#     parser.add_argument("json_final", type=str)
#     args = parser.parse_args()

#     cfg = load_config(path_from_file(__file__, "frame_sampling_config.yaml"))
#     random.seed(cfg['settings'].get('seed', 123456789))

#     data = join_dataframe_dict(load_json(args.in_json_path))
#     mode = cfg['settings'].get('selection_mode', 'many')
#     use_ca = cfg['settings'].get('use_ca_score', True)
#     use_vp = cfg['settings'].get('use_viewpoint', True)

#     print(f"Sampling tracks using '{mode}' mode...")
#     final_list = run_many_mode_logic(data, cfg) if mode == 'many' else run_one_mode_logic(data, use_ca, use_vp)

#     print(f"Saving {len(final_list)} annotations...")
#     save_json(split_dataframe(pd.DataFrame(final_list)), args.json_final)

# def run_one_mode_logic(data, use_ca, use_vp):
#     processed = []
#     # Standard one-best selection
#     by_vp_tid = defaultdict(lambda: defaultdict(list))
#     for ann in data['annotations']:
#         vp = ann.get('viewpoint', 'unknown') if use_vp else 'unknown'
#         by_vp_tid[vp][ann['tracking_id']].append(ann)
#     for vp, tracks in by_vp_tid.items():
#         for tid, anns in tracks.items():
#             anns.sort(key=lambda a: (a.get('CA_score', 0), -a['frame_number']), reverse=True)
#             processed.append(anns[0])
#     return processed

# if __name__ == "__main__":
#     main()

def run_one_mode_logic(data, use_ca, use_vp):
    processed = []
    # Standard one-best selection
    by_vp_tid = defaultdict(lambda: defaultdict(list))
    for ann in data['annotations']:
        vp = ann.get('viewpoint', 'unknown') if use_vp else 'unknown'
        by_vp_tid[vp][ann['tracking_id']].append(ann)
    for vp, tracks in by_vp_tid.items():
        for tid, anns in tracks.items():
            anns.sort(key=lambda a: (a.get('CA_score', 0), -a['frame_number']), reverse=True)
            processed.append(anns[0])
    return processed

def main():
    parser = argparse.ArgumentParser(description="KABR Unified Sampling - Final")
    parser.add_argument("in_json_path", type=str)
    parser.add_argument("json_final", type=str)

    # Backward-compatible: old pipeline may still pass this
    parser.add_argument(
        "--json_stage1",
        type=str,
        default=None,
        help="Backward-compatible argument (optional). If provided, saves a copy of the output here."
    )

    args = parser.parse_args()

    cfg = load_config(path_from_file(__file__, "frame_sampling_config.yaml"))
    random.seed(cfg['settings'].get('seed', 123456789))

    data = join_dataframe_dict(load_json(args.in_json_path))
    mode = cfg['settings'].get('selection_mode', 'many')
    use_ca = cfg['settings'].get('use_ca_score', True)
    use_vp = cfg['settings'].get('use_viewpoint', True)

    print(f"Sampling tracks using '{mode}' mode...")
    final_list = run_many_mode_logic(data, cfg) if mode == 'many' else run_one_mode_logic(data, use_ca, use_vp)

    # Save final output
    print(f"Saving {len(final_list)} annotations to {args.json_final}...")
    final_data = split_dataframe(pd.DataFrame(final_list))
    save_json(final_data, args.json_final)

    # If stage1 path is provided (old workflow), save a copy for compatibility
    if args.json_stage1:
        print(f"(Compat) Also saving stage1 copy to {args.json_stage1}...")
        save_json(final_data, args.json_stage1)

if __name__ == "__main__":
    main()
