import json
from collections import defaultdict
from copy import deepcopy
import yaml
import random
import argparse  # Added for command line arguments


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


random.seed(123456789)  # for reproducibility


def group_annotations_by_tracking_id_and_subsequences(data):
    """
    Groups annotations by tracking ID and splits them into maximal
    subsequences of consecutive frames.
    """
    annotations = data["annotations"]
    images = {image["uuid"]: image for image in data["images"]}
    tracking_id_annotations = defaultdict(list)

    for annotation in annotations:
        image_uuid = annotation["image_uuid"]
        if image_uuid in images:
            image_path = images[image_uuid]["image path"]
            # Extract frame number from file name (assumes format ending with _<frame>.ext)
            frame_number_str = image_path.split("_")[-1].split(".")[0]
            try:
                frame_number = int(frame_number_str)
            except ValueError:
                print(frame_number_str)
                print(
                    f"Warning: Could not extract frame number from file name {image_path}"
                )
                continue
            annotation["frame_number"] = frame_number
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
    data,
    # t_seconds,
    # frame_interval,
    frames_per_subsequence,
    ca_available,
    viewpoint_available,
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
        viewpoint = (
            annotation.get("viewpoint", "unknown") if viewpoint_available else "unknown"
        )
        annotations_by_viewpoint[viewpoint].append(annotation)

    filtered_annotations = []
    # Convert time gap (seconds) to frames
    # t_frames = t_seconds * frame_intervals

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

                if ca_available:
                    subseq_sorted_by_ca = sorted(
                        subseq, key=lambda x: x.get("CA_score", -1), reverse=True
                    )
                    selected_subseq_annotations = subseq_sorted_by_ca[
                        :frames_per_subsequence
                    ]
                    for ann in selected_subseq_annotations:
                        print(
                            f"      Selected annotation at frame {ann['frame_number']} (CA: {ann.get('CA_score', 'N/A')})"
                        )

                else:
                    selected_subseq_annotations = random.sample(
                        subseq, min(len(subseq), frames_per_subsequence)
                    )
                    for ann in selected_subseq_annotations:
                        print(
                            f"      Randomly selected annotation at frame {ann['frame_number']}"
                        )

                filtered_annotations.extend(selected_subseq_annotations)

    filtered_size = len(filtered_annotations)
    print(f"\nNumber of annotations after Stage 1 filtering: {filtered_size}")
    data_copy["annotations"] = filtered_annotations
    return data_copy


def group_annotations_by_tracking_id_and_viewpoint(data, viewpoint_available):
    """
    Groups annotations by tracking ID for the expected viewpoints ('left' and 'right').
    """
    if viewpoint_available:
        annotations_by_viewpoint = {
            "left": defaultdict(list),
            "right": defaultdict(list),
        }
        for annotation in data["annotations"]:
            viewpoint = annotation.get("viewpoint")
            if viewpoint not in annotations_by_viewpoint:
                print(
                    f"Unknown viewpoint '{viewpoint}' encountered. Skipping annotation."
                )
                continue
            tracking_id = annotation.get("tracking_id", "N/A")
            annotations_by_viewpoint[viewpoint][tracking_id].append(annotation)
    else:
        annotations_by_viewpoint = {"unknown": defaultdict(list)}
        for annotation in data["annotations"]:
            tracking_id = annotation.get("tracking_id", "N/A")
            annotations_by_viewpoint["unknown"][tracking_id].append(annotation)
    return annotations_by_viewpoint


def filter_annotations(annotations_by_viewpoint, threshold_percentage, ca_available):
    """
    Stage 2: For each tracking ID (per viewpoint), filters annotations to keep only those
    that have a CA score above a threshold (percentage of the highest CA score).
    Prints counts before and after filtering.
    """
    for viewpoint, annotations_by_tracking_id in annotations_by_viewpoint.items():
        for tracking_id, annotations in annotations_by_tracking_id.items():

            if ca_available:
                highest_ca_score = max(
                    annotation.get("CA_score", 0) for annotation in annotations
                )
                threshold = threshold_percentage * highest_ca_score
                filtered_annotations = [
                    annotation
                    for annotation in annotations
                    if annotation.get("CA_score", 0) >= threshold
                ]

                print(f"\nViewpoint '{viewpoint}', Tracking ID '{tracking_id}':")
                print(f"  Highest CA score: {highest_ca_score}")
                print(f"  Threshold ({threshold_percentage*100:.0f}%): {threshold}")
            else:
                num_to_keep = max(1, int(len(annotations) * threshold_percentage))
                filtered_annotations = random.sample(annotations, num_to_keep)
                print(f"\nViewpoint '{viewpoint}', Tracking ID '{tracking_id}':")
                print(
                    f"  Random selection - keeping {threshold_percentage*100:.0f}% of annotations"
                )

            annotations_by_tracking_id[tracking_id] = filtered_annotations
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


def ensure_time_separation(data, t_seconds, frame_interval, ca_available):
    """
    Stage 3: Ensures that for each tracking ID, annotations are at least t_seconds apart.
    If annotations are closer than this threshold, keeps only the one with highest CA score.
    """
    annotations = data["annotations"]

    # Group annotations by tracking ID and viewpoint if available
    tracking_id_viewpoint_annotations = defaultdict(list)
    for annotation in annotations:
        tracking_id = annotation.get("tracking_id", "unknown")
        viewpoint = annotation.get("viewpoint", "unknown")
        key = (tracking_id, viewpoint)  # Use a tuple as the key
        tracking_id_viewpoint_annotations[key].append(annotation)

    # Calculate frame threshold (how many frames correspond to t_seconds)
    frame_threshold = max(1, int(t_seconds * frame_interval))
    print(f"\n=== Starting Stage 3: Ensuring Time Separation ===")
    print(
        f"Ensuring annotations are at least {t_seconds} seconds apart ({frame_threshold} frames)"
    )

    final_annotations = []

    for key, anns in tracking_id_viewpoint_annotations.items():
        tracking_id, viewpoint = key
        print(f"\nProcessing Tracking ID '{tracking_id}', Viewpoint '{viewpoint}':")
        print(f"  Initial annotations: {len(anns)}")

        # Sort annotations by frame number
        anns_sorted = sorted(anns, key=lambda x: x.get("frame_number", 0))

        # Process sorted annotations to ensure time separation
        kept_annotations = []
        i = 0

        while i < len(anns_sorted):
            current_ann = anns_sorted[i]
            current_frame = current_ann.get("frame_number", 0)

            # Find all annotations that are too close to the current one
            close_annotations = [current_ann]
            j = i + 1
            while j < len(anns_sorted):
                next_ann = anns_sorted[j]
                next_frame = next_ann.get("frame_number", 0)

                if next_frame - current_frame < frame_threshold:
                    # This annotation is too close
                    close_annotations.append(next_ann)
                    j += 1
                else:
                    # Found an annotation that's far enough
                    break

            # From the close annotations, select the one with highest CA score
            if ca_available and len(close_annotations) > 1:
                best_ann = max(close_annotations, key=lambda x: x.get("CA_score", 0))
                ca_score = best_ann.get("CA_score", 0)
                print(
                    f"  From frames {close_annotations[0]['frame_number']} to {close_annotations[-1]['frame_number']}, "
                    f"selected frame {best_ann['frame_number']} with CA score {ca_score}"
                )
            else:
                # If CA scores not available, just take the first annotation
                best_ann = close_annotations[0]
                if len(close_annotations) > 1:
                    print(
                        f"  From frames {close_annotations[0]['frame_number']} to {close_annotations[-1]['frame_number']}, "
                        f"selected first frame {best_ann['frame_number']}"
                    )
                else:
                    print(f"  Kept lone annotation at frame {best_ann['frame_number']}")

            kept_annotations.append(best_ann)

            # Move to the next block of annotations
            i = j

        print(f"  Final annotations after time separation: {len(kept_annotations)}")
        final_annotations.extend(kept_annotations)

    print(
        f"\nTotal annotations after ensuring time separation: {len(final_annotations)}"
    )
    data["annotations"] = final_annotations
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Run frame sampling algorithm on ca classifier output"
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

    # Set defaults based on original behavior
    parser.set_defaults(viewpoint=False, ca_score=False)

    args = parser.parse_args()

    # Load configuration from YAML file
    config = load_config("algo/frame_sampling.yaml")
    input_file = args.in_json_path
    stage1_output = args.json_stage1
    final_output = args.json_final

    t_seconds = config["thresholds"]["t_seconds"]
    frame_interval = config["thresholds"]["frame_interval"]
    threshold_percentage_stage2 = config["thresholds"]["threshold_percentage_stage2"]

    # Override config values with command line arguments
    viewpoint_available = config["settings"]["use_viewpoint"]
    ca_available = config["settings"]["use_ca_score"]

    print(f"Using viewpoint processing: {viewpoint_available}")
    print(f"Using CA score processing: {ca_available}")

    print("=== Starting Stage 1: Frame Sampling Algorithm ===")
    data = load_json(input_file)
    processed_data = frame_sampling_algorithm_combined(
        data,
        # t_seconds,
        # frame_interval,
        frames_per_subsequence=config["thresholds"]["frames_per_subsequence"],
        ca_available=False,
        viewpoint_available=False,
    )
    save_json(processed_data, stage1_output)
    print(f"Stage 1 output saved to {stage1_output}")

    print("\n=== Starting Stage 2: CA Score Threshold Filtering ===")
    data_stage1 = load_json(stage1_output)
    initial_stage2_count = len(data_stage1["annotations"])
    print(f"Number of annotations at start of Stage 2: {initial_stage2_count}")

    annotations_by_viewpoint = group_annotations_by_tracking_id_and_viewpoint(
        data_stage1, viewpoint_available
    )
    filtered_annotations_by_viewpoint = filter_annotations(
        annotations_by_viewpoint,
        threshold_percentage_stage2,
        ca_available,
    )
    final_data = reconstruct_annotations(data_stage1, filtered_annotations_by_viewpoint)

    final_count = len(final_data["annotations"])
    print(f"\nFinal number of annotations after Stage 2 filtering: {final_count}")

    # Add Stage 3: Ensure time separation between annotations
    final_data_with_separation = ensure_time_separation(
        final_data, t_seconds, frame_interval, ca_available
    )

    save_json(final_data_with_separation, final_output)
    print(f"Final output saved to {final_output}")


if __name__ == "__main__":
    main()
