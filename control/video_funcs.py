import shutil
import cv2
from os.path import exists, isabs, join
from pathlib import Path
from PIL import Image
import os

import constants as const
import utils as ut
from algo import preproc
from control.ggr_funcs import extrapolate_ggr_gps
import control.ggr_funcs as ggr

def process_video_by_frame(cap, in_file, out_file):
    """
    Processes a video frame-by-frame while saving to a new location.
    """
    frame_dims = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    codec = cv2.VideoWriter_fourcc(*'MP4V')

    writer = cv2.VideoWriter(out_file,codec,fps,frame_dims)
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        # If frame is read correctly ret is True, if False we finished
        if ret:
            # TODO: FRAME BY FRAME PROCESSING HAPPENS HERE
            # (process bit depth) ...

            # Save frame by frame into the writer
            writer.write(frame)
        else:
            print(f"[pipeline] Video at {in_file} has been processed and copied into {out_file}.")
            break
        
    # Release writer but not capture as it may still be in use
    writer.release()


def add_videos(
    dir_out,
    gpath_list,
    frame_rate=8,
    auto_localize=True,
    location_for_names="GGR",
    ensure_loadable=True,
    doctest_mode=False,
):
    """
    Adds a list of video paths to the image table.

    Initially we set the video_uri to exactly the given gpath.
    Later we change the uri, but keeping it the same here lets
    us process images asychronously.

    Parameters:
        dir_out (str): directory to load videos into
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

    vid_dir = os.path.join(dir_out,"videos")
    trash_dir = os.path.join(dir_out,"trash")

    # Create database directory if it doesn't exist
    Path(vid_dir).mkdir(parents=True, exist_ok=True)
    Path(trash_dir).mkdir(parents=True, exist_ok=True)

    # Check loadable
    if ensure_loadable:
        for g in gpath_list:
            
            sep = g.rfind("/")
            ext = g.rfind(".")
            fname = g[sep:ext].replace("/","")
            fname_avi = fname + ".mp4"
            vid_dest = os.path.join(vid_dir,fname_avi)
            trash_dest = os.path.join(trash_dir,fname_avi)

            try:
                v = cv2.VideoCapture(g)

                if not v.isOpened():
                    print(f"[pipeline] Video failed to open: {g}")

                    shutil.copy2(g, trash_dest)
                    print(f"[pipeline] Video {g} has been copied into {trash_dest}.")
                else:
                    # PROCESS VIDEOS AND SAVE TO NEW LOCATION
                    process_video_by_frame(v,g,vid_dest)
                    v.release()
                    
            except Exception as e:
                print(f"[pipeline] Error loading video: {g}")

                shutil.copy2(g, trash_dest)
                print(f"[pipeline] Video {g} has been copied into {trash_dest}.")
