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


def add_videos(
    dir_out,
    gpath_list,
    frame_rate=8,
    max_frames=300,
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

    Doctest Command:
        python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE control/image_funcs.py
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
                        "frame data": vid_params,
                    })
                    v.release()
                    
            except Exception as e:
                print(f"[pipeline] Error loading video: {g}")

                shutil.copy2(g, trash_dest)
                print(f"[pipeline] Video {g} has been copied into {trash_dest}.")

    return {"videos": video_params}