import argparse
import cv2
import os
import ultralytics
import uuid
import warnings

import pandas as pd

from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from VAREID.libraries.io.format_funcs import load_config, load_json, save_json, split_dataframe
from VAREID.libraries.utils import path_from_file

ultralytics.checks()
warnings.filterwarnings("ignore")


def detect_videos(video_data, model_path, threshold, sz):
    """Run YOLO detection + ByteTrack on a single video's frames."""
    videos = video_data["videos"]
    annotations = []

    for vid in videos:
        vid_name = vid["video fname"]
        frames = vid["frame data"]

        # MAKE A NEW YOLO MODEL FOR EACH VIDEO
        model = YOLO(model_path)

        # DETECT AND TRACK OVER VIDEO
        for i, frame_data in enumerate(tqdm(frames, desc=f"Detecting frames from {vid_name}...")):
            img = cv2.imread(frame_data["uri"])
            
            results = model.track(img, verbose=False, persist=True, imgsz=sz)

            # Extract detections and tracking information
            for result in results:
                # Check if any detection in the image is a person (class 0)
                if any(box.cls.item() == 0 for box in result.boxes):
                    # Skip this entire image
                    continue
                
                # Iterate over detections in frame and only accept those above threshold
                for box in result.boxes:
                    if box.conf is None or box.conf.item() < threshold:
                        continue

                    x1, y1, x2, y2 = (box.xyxy[0][0].item(), box.xyxy[0][1].item(), box.xyxy[0][2].item(), box.xyxy[0][3].item())

                    annotations.append({
                        "uuid": str(uuid.uuid4()),
                        "image_uuid": frame_data["uuid"],
                        "image_path": frame_data["uri"],
                        "video_path": vid["video path"],
                        "frame_number": i + 1,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "confidence": box.conf.item() if box.conf is not None else -1,
                        "detection_class": int(box.cls.item()) if box.cls is not None else -1,
                        "tracking_id": int(box.id.item()) if box.id is not None else -1,
                        "timestamp": frame_data["time_posix"],
                    })
        
        print(f"Finished detecting frames from {vid_name}.")

    print(f"Finished all detecting!")
    return annotations


def postprocess_tracking_ids(annots):
    """
    Ensures that tracking ids for separate videos do not overlap in the same range of numbers
    by remapping tracking ids to unused integer values.
    """

    # Defines which video paths use which tracking id. tracking id -> video path
    used_keys = {}
    # Defines mappings to follow. (video path, tracking id) -> new tracking id
    mappings = {}
    # The smallest unused tracking id
    next_unused_id = 1

    for index, annot in enumerate(annots):
        tid = annot["tracking_id"]
        path = annot["video_path"]

        # Check if the key is used by a different image
        if tid in used_keys.keys() and used_keys[tid] != path:
            # If it is, check if a mapping exists yet
            mapping_key = (path, tid)
            if mapping_key in mappings.keys():
                tid = mappings[mapping_key]
            # If it doesn't create the mapping
            else:
                mappings[mapping_key] = next_unused_id
                tid = next_unused_id

        # Mark the new key if needed
        if tid not in used_keys.keys():
            used_keys[tid] = path
            # Find the next unused id not in used_keys
            while next_unused_id in used_keys.keys():
                next_unused_id += 1

        # Assign the tid
        annots[index]["tracking_id"] = tid


def main(args):
    config = load_config(path_from_file(__file__, "detector_config.yaml"))
    
    dt_dir = Path(args.dt_dir)
    video_data = load_json(args.video_data)
    os.makedirs(dt_dir, exist_ok=True)

    threshold = config["confidence_threshold"]
    sz = config["img_size_vid"]
    
    all_annotations = detect_videos(video_data, args.model_path, threshold, sz)

    print("Post-processing tracking IDs to avoid collisions…")
    postprocess_tracking_ids(all_annotations)

    df = pd.DataFrame(all_annotations)
    annotations_final = split_dataframe(df)

    out_json = os.path.join(dt_dir, args.out_json_path)
    print(f"Saving annotations to {out_json}…")
    save_json(annotations_final, out_json)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect and track bounding boxes for a database of animal videos",
    )
    parser.add_argument("video_data", type=str, help="The video metadata file")
    parser.add_argument("dt_dir", type=str, help="Directory to export models and annots to")
    parser.add_argument("model_path", type=str, help="YOLO model path (or path to create)")
    parser.add_argument("out_json_path", type=str, help="Name of the output annotations JSON")

    args = parser.parse_args()
    main(args)