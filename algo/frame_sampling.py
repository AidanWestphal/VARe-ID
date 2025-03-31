import json
from collections import defaultdict
from copy import deepcopy
import yaml
import argparse


def load_json(file_path):
    """Load JSON data from the given file."""
    with open(file_path, "r") as file:
        return json.load(file)


def save_json(data, file_path):
    """Save JSON data to the given file."""
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


def load_config(config_path):
    """Load YAML configuration from the given file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def group_annotations_by_tracking_id_and_subsequences(data):
    """
    Groups annotations by tracking ID and splits them into maximal
    subsequences of consecutive frames.
    """
    annotations = data["annotations"]
    tracking_id_annotations = defaultdict(list)

    for annotation in annotations:
        tracking_id = annotation["tracking_id"]
        tracking_id_annotations[tracking_id].append(annotation)

    tracking_id_subsequences = {}
    for tracking_id, anns in tracking_id_annotations.items():
        anns_sorted = sorted(anns, key=lambda x: x["frame_number"])
        subsequences = []
        current_subseq = []
        previous_frame_number = None

        for ann in anns_sorted:
            frame_number = ann["frame_number"]
            if (
                previous_frame_number is None
                or frame_number == previous_frame_number + 1
            ):
                current_subseq.append(ann)
            else:
                if current_subseq:
                    subsequences.append(current_subseq)
                current_subseq = [ann]
            previous_frame_number = frame_number

        if current_subseq:
            subsequences.append(current_subseq)

        tracking_id_subsequences[tracking_id] = subsequences

    return tracking_id_subsequences


def frame_sampling_algorithm_combined(
    data, t_seconds, frame_interval, threshold_percentage
):
    """
    Stage 1: Applies frame sampling on the data.
    Prints initial annotation count, details per viewpoint, per tracking ID, and per subsequence.
    Returns the processed JSON data.
    """
    data_copy = deepcopy(data)
    original_size = len(data_copy["annotations"])
    print(f"\nInitial number of annotations: {original_size}")

    # Separate annotations by viewpoint
    annotations_by_viewpoint = defaultdict(list)
    for annotation in data_copy["annotations"]:
        viewpoint = annotation.get("viewpoint", "unknown")
        annotations_by_viewpoint[viewpoint].append(annotation)

    filtered_annotations = []
    # Convert time gap (seconds) to frames
    t_frames = t_seconds * frame_interval

    for viewpoint, annotations in annotations_by_viewpoint.items():
        print(f"\nProcessing Viewpoint '{viewpoint}':")
        temp_data = {"images": data["images"], "annotations": annotations}
        tracking_id_subsequences = group_annotations_by_tracking_id_and_subsequences(
            temp_data
        )
        for tracking_id, subsequences in tracking_id_subsequences.items():
            print(f"\n  Processing Tracking ID {tracking_id}:")
            for idx, subseq in enumerate(subsequences):
                print(f"    Subsequence {idx+1} with {len(subseq)} annotations")
                max_ca_annotation = max(subseq, key=lambda x: x["CA_score"])
                max_ca_score = max_ca_annotation["CA_score"]
                threshold = max_ca_score * threshold_percentage
                print(f"      Max CA score: {max_ca_score}")
                print(
                    f"      Threshold ({threshold_percentage*100:.0f}% of max): {threshold}"
                )

                # Always select the annotation with the maximum CA score
                selected_subseq_annotations = [max_ca_annotation]
                selected_frames = [max_ca_annotation["frame_number"]]
                print(
                    f"      Selected annotation at frame {max_ca_annotation['frame_number']} (Max CA)"
                )

                subseq_sorted = sorted(subseq, key=lambda x: x["frame_number"])
                # Look for local maxima within the subsequence
                for i in range(1, len(subseq_sorted) - 1):
                    ann_prev = subseq_sorted[i - 1]
                    ann_current = subseq_sorted[i]
                    ann_next = subseq_sorted[i + 1]

                    if (
                        ann_current["CA_score"] >= ann_prev["CA_score"]
                        and ann_current["CA_score"] >= ann_next["CA_score"]
                        and ann_current["CA_score"] >= threshold
                    ):

                        # Ensure a minimum time gap between selected annotations
                        far_enough = all(
                            abs(ann_current["frame_number"] - f) >= t_frames
                            for f in selected_frames
                        )
                        if far_enough:
                            selected_subseq_annotations.append(ann_current)
                            selected_frames.append(ann_current["frame_number"])
                            print(
                                f"      Selected annotation at frame {ann_current['frame_number']} (Local Max)"
                            )

                filtered_annotations.extend(selected_subseq_annotations)

    filtered_size = len(filtered_annotations)
    print(f"\nNumber of annotations after Stage 1 filtering: {filtered_size}")
    data_copy["annotations"] = filtered_annotations
    return data_copy


def group_annotations_by_tracking_id_and_viewpoint(data):
    """
    Groups annotations by tracking ID for the expected viewpoints ('left' and 'right').
    """
    annotations_by_viewpoint = {"left": defaultdict(list), "right": defaultdict(list)}
    for annotation in data["annotations"]:
        viewpoint = annotation.get("viewpoint")
        if viewpoint not in annotations_by_viewpoint:
            print(f"Unknown viewpoint '{viewpoint}' encountered. Skipping annotation.")
            continue
        tracking_id = annotation.get("tracking_id", "N/A")
        annotations_by_viewpoint[viewpoint][tracking_id].append(annotation)
    return annotations_by_viewpoint


def filter_annotations(annotations_by_viewpoint, threshold_percentage):
    """
    Stage 2: For each tracking ID (per viewpoint), filters annotations to keep only those
    that have a CA score above a threshold (percentage of the highest CA score).
    Prints counts before and after filtering.
    """
    for viewpoint, annotations_by_tracking_id in annotations_by_viewpoint.items():
        for tracking_id, annotations in annotations_by_tracking_id.items():
            highest_ca_score = max(
                annotation.get("CA_score", 0) for annotation in annotations
            )
            threshold = threshold_percentage * highest_ca_score
            filtered_annotations = [
                annotation
                for annotation in annotations
                if annotation.get("CA_score", 0) >= threshold
            ]
            annotations_by_tracking_id[tracking_id] = filtered_annotations
            print(f"\nViewpoint '{viewpoint}', Tracking ID '{tracking_id}':")
            print(f"  Highest CA score: {highest_ca_score}")
            print(f"  Threshold ({threshold_percentage*100:.0f}%): {threshold}")
            print(
                f"  Annotations before filtering: {len(annotations)}, after filtering: {len(filtered_annotations)}"
            )
    return annotations_by_viewpoint


def reconstruct_annotations(data, annotations_by_viewpoint):
    """
    Reconstructs the JSON annotations list from the grouped annotations.
    """
    new_annotations = []
    for viewpoint, annotations_by_tracking_id in annotations_by_viewpoint.items():
        for tracking_id, annotations in annotations_by_tracking_id.items():
            new_annotations.extend(annotations)
    data["annotations"] = new_annotations
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Run frame sampling algorithm on ca classifier output"
    )
    parser.add_argument(
        "image_dir", type=str, help="The directory where localized images are found"
    )
    parser.add_argument(
        "in_json_path",
        type=str,
        help="The full path to the ca classifier output json to use as input",
    )
    parser.add_argument(
        "json_stage1",
        type=str,
        help="The full path to the output json file for stage 1",
    )
    parser.add_argument(
        "json_final", type=str, help="The full path to the output json file for final"
    )
    args = parser.parse_args()

    input_file = args.in_json_path
    stage1_output = args.json_stage1
    final_output = args.json_final

    # Load configuration from YAML file
    config = load_config("algo/frame_sampling.yaml")

    t_seconds = config["thresholds"]["t_seconds"]
    frame_interval = config["thresholds"]["frame_interval"]
    threshold_percentage_stage1 = config["thresholds"]["threshold_percentage_stage1"]
    threshold_percentage_stage2 = config["thresholds"]["threshold_percentage_stage2"]

    print("=== Starting Stage 1: Frame Sampling Algorithm ===")
    data = load_json(input_file)
    processed_data = frame_sampling_algorithm_combined(
        data, t_seconds, frame_interval, threshold_percentage_stage1
    )
    save_json(processed_data, stage1_output)
    print(f"Stage 1 output saved to {stage1_output}")

    print("\n=== Starting Stage 2: CA Score Threshold Filtering ===")
    data_stage1 = load_json(stage1_output)
    initial_stage2_count = len(data_stage1["annotations"])
    print(f"Number of annotations at start of Stage 2: {initial_stage2_count}")

    annotations_by_viewpoint = group_annotations_by_tracking_id_and_viewpoint(
        data_stage1
    )
    filtered_annotations_by_viewpoint = filter_annotations(
        annotations_by_viewpoint, threshold_percentage_stage2
    )
    final_data = reconstruct_annotations(data_stage1, filtered_annotations_by_viewpoint)

    final_count = len(final_data["annotations"])
    print(f"\nFinal number of annotations after Stage 2 filtering: {final_count}")

    save_json(final_data, final_output)
    print(f"Final output saved to {final_output}")


if __name__ == "__main__":
    main()
