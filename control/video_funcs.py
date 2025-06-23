from datetime import datetime
import re
import shutil
import cv2
from pathlib import Path
import os
from algo.preproc import parse_imageinfo
import constants as const


EXIF_NORMAL = const.ORIENTATION_DICT_INVERSE[const.ORIENTATION_000]
EXIF_UNDEFINED = const.ORIENTATION_DICT_INVERSE[const.ORIENTATION_UNDEFINED]
IMAGE_COLNAMES = (
    "uuid",
    "uri",
    "uri_original",
    "original_name",
    "ext",
    "width",
    "height",
    "time_posix",
    "gps_lat",
    "gps_lon",
    "orientation",
    "note",
)


def process_video_by_frame(cap, file_name, img_dir, frame_rate=8, max_frames=2000):
    """
    Processes and split a video frame-by-frame while saving to a new location.
    """
    # frame_dims = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    max_num_length = len(str(max_frames))

    dot = file_name.rfind(".")
    vid_name = file_name[:dot]

    original_frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_frame_rate
    print(f"[pipeline] Original frame rate: {original_frame_rate}")
    print(f"[pipeline] Total frames in the video: {total_frames}")
    print(f"[pipeline] Video duration: {duration} seconds")

    frame_interval = round(original_frame_rate / frame_rate)
    print(f"[pipeline] Frame interval for extraction: {frame_interval}")

    # codec = cv2.VideoWriter_fourcc(*'MP4V')
    # writer = cv2.VideoWriter(out_file,codec,frame_rate,frame_dims)

    extracted_frames = 0
    current_frame = 0
    params_list = []

    while extracted_frames < max_frames:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break
        
        if current_frame % frame_interval == 0:
            # TODO: PROCESS FRAME FOR BIT DEPTH, ETC.
            # writer.write(frame)
            extracted_frames += 1

            f_name = vid_name + "_" + str(extracted_frames).zfill(max_num_length) + ".jpg"
            f_path = os.path.join(img_dir,f_name)
            cv2.imwrite(f_path,frame)

            img_info = parse_imageinfo(f_path)
            params_list.append({
                key: value for key, value in zip(IMAGE_COLNAMES, img_info)
            })
        
        current_frame += 1
          
    # Release writer but not capture as it may still be in use
    # writer.release()
    print(f"[pipeline] Video {file_name} processed. Total frames extracted: {extracted_frames}")
    return params_list


def link_srts(video_data, srts):
    """
    Links SRT files to their corresponding videos in a video data JSON file.

    We assume the SRT is located in the same directory as the video and named the same 
    way as the video (with the exception of the file extension).

    This function directly modifies the provided video_data dictionary.

    Parameters:
        video_data (dict): The JSON formatted video data. This data field is MODIFIED.
        srts (Path): The path object containing all SRT files.

    Returns:
        video_data (dict): The modified JSON formatted video data.
    """

    print(f"[pipeline] link_srts")
    for srt in srts:
        ext = srt.rfind(".")
        srt_path = srt[:ext]
        # Find matching videos
        for index, video in enumerate(video_data["videos"]):
            # (for logging purposes)
            vp = video["video path"]
            v_ext = video["video path"].rfind(".")
            vid_path = video["video path"][:v_ext]            
            # Match the SRT file iff the video shares the same name and is in the same directory
            if vid_path == srt_path:
                video_data["videos"][index]["srt path"] = srt
                # Files in same dir can't have duplicate names, assume we can break here
                print(f"[pipeline] Video {vp} linked to SRT file {srt}.")
                break

    print(f"[pipeline] SRT files have finished linking.")


def convert_timestamp(timestamp):
    """
    Converts a timestamp from the format "YYYY-MM-DD HH:MM:SS,SSS,SSS" 
    into a posix time. (where SS,SSS,SSS is second, millisecond, and microsecond)
    """

    # Remove last comma, replace second-to-last as a dot
    sep = timestamp.rfind(",")
    timestamp = timestamp[:sep] + timestamp[(sep + 1):]
    sep = timestamp.rfind(",")
    timestamp = timestamp[:sep] + "." + timestamp[(sep + 1):]

    # Data is now in "YYYY-MM-DD HH:MM:SS.SSSSSS" format
    format = "%Y-%m-%d %H:%M:%S.%f"
    dt = datetime.strptime(timestamp,format)

    return dt.timestamp()


def parse_srt(srt_path):
    """
    Parse an SRT file and return a dict mapping:
        srt_dict[SrtCnt] = "timestamp string"
    For example, lines in the SRT might look like:

        SrtCnt : 1, DiffTime : 33ms
        2023-01-19 10:56:36,107,334

    We'll look for `SrtCnt : X` and store the next line
    (assuming it’s the date/time) as srt_dict[X] in posix form.

    In other words, this function returns a mapping from 
    frame number to a timestamp.
    """
    print(f"[pipeline] parse_srt")
    srt_dict = {}
    with open(srt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    # Diff in time (ms), used in last step
    last_dt = 0
    # Srt entry s.t. we can find the next key later
    srt_cnt = 0

    while i < len(lines):
        line = lines[i].strip()

        # Look for something like "SrtCnt : 123"
        match = re.search(r"SrtCnt\s*:\s*(\d+)", line)
        if match:
            # On this line, grab the difftime for later (should ALWAYS exist, but checked just in case)
            dt = re.search(r"DiffTime\s*:\s*(\d+)ms", line)
            if dt:
                last_dt = int(dt.group(1))/1000

            srt_cnt = int(match.group(1))  # This is 1-based
            # The very next line should have the date/time
            if i + 1 < len(lines):
                possible_time = lines[i + 1].strip()
                # If it looks like a datetime, store it
                if re.search(
                    r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3},\d+", possible_time
                ):
                    # Convert timestamp to posix format
                    srt_dict[srt_cnt] = convert_timestamp(possible_time)
            i += 2
        else:
            i += 1

    # Calculate the final entry manually using prior time difference
    srt_dict[srt_cnt + 1] = srt_dict[srt_cnt] + last_dt

    print(f"[pipeline] Finished parsing SRT {srt_path}.")
    return srt_dict


def update_timestamps(video_data, desired_fps):
    """
    Adds timestamp information to the video data metafile dictionary.
    """
    print(f"[pipeline] update_timestamps")
    for index, video in enumerate(video_data["videos"]):
        # Obtain srt and frame interval
        srt_path = video["srt path"]
        srt = parse_srt(srt_path)
        frame_interval = round(video["fps"] / desired_fps)
        # (for logging purposes)
        vp = video["video path"]

        print(f"[pipeline] Assigning timestamps for {vp}.")
        # Iterate over frame data and add timestamps to each
        for frame_number, frame in enumerate(video["frame data"]):
            # Scale by frame interval and add 1-index
            original_frame_number = frame_number * frame_interval + 1
            # Assign timestamp
            frame["time_posix"] = srt[original_frame_number]

        print(f"[pipeline] Finished assigning timestamps for {vp}.")

    print(f"[pipeline] Finished assigning all timestamps.")

def add_videos(
    dir_out,
    gpath_list,
    frame_rate=8,
    max_frames=2000,
    ensure_loadable=True
):
    """
    Adds a list of video paths to the image table.

    Initially we set the video_uri to exactly the given gpath.
    Later we change the uri, but keeping it the same here lets
    us process images asychronously.

    Parameters:
        dir_out (str): directory to load images into
        gpath_list (list): list of video paths to add
        auto_localize (bool): if None uses the default specified in ibs.cfg
        location_for_names (str):
        ensure_loadable (bool): check whether imported images can be loaded.  Defaults to
            True
        doctest_mode (bool): if true, replaces carriage returns with newlines

    Returns:
        gid_list (list of rowids): gids are image rowids
    """

    print(f"[pipeline] add_videos")
    print(f"[pipeline] len(gpath_list) = {len(gpath_list)}")
    if len(gpath_list) == 0:
        print(f"[pipeline] No videos to load: exiting...")
        return []

    img_dir = os.path.join(dir_out,"images")
    trash_dir = os.path.join(dir_out,"trash")

    # Create database directory if it doesn't exist
    Path(img_dir).mkdir(parents=True, exist_ok=True)
    Path(trash_dir).mkdir(parents=True, exist_ok=True)

    video_params = []

    i = 0
    # Check loadable
    if ensure_loadable:
        for g in gpath_list:

            sep = g.rfind("/")
            fname = g[sep:].replace("/","")
            trash_dest = os.path.join(trash_dir,fname)

            try:
                v = cv2.VideoCapture(g)

                if not v.isOpened():
                    print(f"[pipeline] Video failed to open: {g}")

                    shutil.copy2(g, trash_dest)
                    print(f"[pipeline] Video {g} has been copied into {trash_dest}.")
                else:
                    # PROCESS VIDEOS AND SAVE TO NEW LOCATION
                    i += 1
                    vid_params = process_video_by_frame(v,fname,img_dir,frame_rate,max_frames)
                    video_params.append({
                        "video id": i,
                        "video fname": fname,
                        "video path": g,
                        "fps": v.get(cv2.CAP_PROP_FPS),
                        "srt path": None,
                        "frame data": vid_params,
                    })
                    v.release()
                    
            except Exception as e:
                print(f"[pipeline] Error loading video: {g}")

                shutil.copy2(g, trash_dest)
                print(f"[pipeline] Video {g} has been copied into {trash_dest}.")

    return {"videos": video_params}