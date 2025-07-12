import cv2
import datetime
import hashlib
import time
import re
import uuid
from os.path import basename, splitext
from PIL import Image

import algo.util.constants as const
from algo.util.utils import parse_exif

EXIF_NORMAL = const.ORIENTATION_DICT_INVERSE[const.ORIENTATION_000]
EXIF_UNDEFINED = const.ORIENTATION_DICT_INVERSE[const.ORIENTATION_UNDEFINED]


def get_unixtime(exif_dict, default=-1):
    """
    Gets unixtime from datetime exif data if it exists for an image.

    Parameters:
        exif_dict (dict): dictionary with formatted image exif data
        default (int): default unixtime value to return if datetime exif data does not exist

    Returns:
        unixtime (int): time image was taken converted to unixtime

    Doctest Command:
        python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE algo/preproc.py

    Example:
        >>> test_exif_dict = {const.EXIF_TIME: "2024:06:28 17:58:16"}
        >>> get_unixtime(test_exif_dict)
        1719611896
        >>> test_exif_dict = {}
        >>> get_unixtime(test_exif_dict)
        -1
    """

    if const.EXIF_TIME in exif_dict.keys():
        dt = re.split(":| ", exif_dict[const.EXIF_TIME])
        dt = datetime.datetime(
            int(dt[0]), int(dt[1]), int(dt[2]), int(dt[3]), int(dt[4]), int(dt[5])
        )
        unixtime = int(time.mktime(dt.timetuple()))
        return unixtime

    return default


def get_lat_lon(exif_dict, default=(-1, -1)):
    """
    Gets latitude and longitude exif data if it exists for an image.

    Parameters:
        exif_dict (dict): dictionary with formatted image exif data
        default (int, int): default latitude and longitude tuple to return if gps exif data does not exist

    Returns:
        gps (tuple): (latitude, longitude) of image converted to decimal degrees

    Doctest Command:
        python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE algo/preproc.py

    Example:
        >>> test_exif_dict = {const.EXIF_LAT: (0, 17, 30.786), const.EXIF_LON: (36, 53, 53.4762)}
        >>> get_lat_lon(test_exif_dict)
        (0.291885, 36.89818783333333)
        >>> test_exif_dict = {const.EXIF_LAT: (0, 17, 30.786)}
        >>> get_lat_lon(test_exif_dict)
        (-1, -1)
        >>> test_exif_dict = {const.EXIF_LON: (36, 53, 53.4762)}
        >>> get_lat_lon(test_exif_dict)
        (-1, -1)
        >>> test_exif_dict = {}
        >>> get_lat_lon(test_exif_dict)
        (-1, -1)
    """

    if const.EXIF_LAT in exif_dict.keys() and const.EXIF_LON in exif_dict.keys():
        lat_tup = exif_dict[const.EXIF_LAT]
        lat = lat_tup[0] + lat_tup[1] / 60 + lat_tup[2] / 3600
        if (
            const.EXIF_LAT_REF in exif_dict.keys()
            and exif_dict[const.EXIF_LAT_REF] == "S"
        ):
            lat = -lat

        lon_tup = exif_dict[const.EXIF_LON]
        lon = lon_tup[0] + lon_tup[1] / 60 + lon_tup[2] / 3600
        if (
            const.EXIF_LON_REF in exif_dict.keys()
            and exif_dict[const.EXIF_LON_REF] == "W"
        ):
            lon = -lon

        return (float(lat), float(lon))

    return default


def get_orientation(exif_dict, default=0):
    """
    Gets orientation exif data if it exists for an image.

    Parameters:
        exif_dict (dict): dictionary with formatted image exif data
        default (int): default orientation value to return if orientation exif data does not exist

    Returns:
        orientation (int): image orientation

    Doctest Command:
        python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE algo/preproc.py

    Example:
        >>> test_exif_dict = {const.EXIF_ORIENT: 1}
        >>> get_orientation(test_exif_dict)
        1
        >>> test_exif_dict = {}
        >>> get_orientation(test_exif_dict)
        0
    """

    return (
        exif_dict[const.EXIF_ORIENT]
        if const.EXIF_ORIENT in exif_dict.keys()
        else default
    )


def get_dim(exif_dict, gpath):
    """
    Gets dimension exif data from image.
    Uses cv2 if PIL cannot extract dimensions

    Parameters:
        exif_dict (dict): dictionary with formatted image exif data
        gpath (str): path to image

    Returns:
        height (int): image height
        width (int): image width

    Doctest Command:
        python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE algo/preproc.py

    Example:
        >>> import os
        >>> import numpy
        >>> import shutil
        >>> from PIL import Image
        >>> db_path = "doctest_files/"
        >>> os.makedirs(db_path + "test_dataset/images")
        >>> a = numpy.random.rand(30,40,3) * 255
        >>> img = Image.fromarray(a.astype('uint8')).convert('RGB')
        >>> img.save(db_path + 'test_dataset/images/img0.JPG')
        >>> test_exif_dict = {const.EXIF_HEIGHT: 30, const.EXIF_WIDTH: 40}
        >>> get_dim(test_exif_dict, db_path + "test_dataset/images/img0.JPG")
        (30, 40)
        >>> test_exif_dict = {}
        >>> get_dim(test_exif_dict, db_path + "test_dataset/images/img0.JPG")
        (30, 40)
        >>> shutil.rmtree(db_path + "test_dataset")
    """

    if (
        const.EXIF_HEIGHT not in exif_dict.keys()
        or const.EXIF_WIDTH not in exif_dict.keys()
    ):
        img = cv2.imread(gpath)
        height, width = img.shape[:2]
        return height, width

    return exif_dict[const.EXIF_HEIGHT], exif_dict[const.EXIF_WIDTH]


def get_exif(pil_img, gpath):
    """
    Gets needed exif fields from image

    Parameters:
        pil_img (open PIL image): PIL image to extract exif data from
        gpath (str): path to pil_img

    Returns:
        unixtime (int): time image was taken converted to unixtime
        lat (float): image latitude converted to decimal degrees
        lon (float): image longitude converted to decimal degrees
        orient (int): image orientation
        height (int): image height
        width (int): image width

    Doctest Command:
        python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE algo/preproc.py

    Example:
        >>> import os
        >>> import numpy
        >>> import exif
        >>> import shutil
        >>> from PIL import Image
        >>> db_path = "doctest_files/"
        >>> gpath = db_path + 'test_dataset/images/img0.JPG'
        >>> os.makedirs(db_path + "test_dataset/images")
        >>> a = numpy.random.rand(30,40,3) * 255
        >>> img = Image.fromarray(a.astype('uint8')).convert('RGB')
        >>> img.save(gpath)
        >>> pil_img = Image.open(gpath, 'r')
        >>> get_exif(pil_img, gpath)
        (-1, -1, -1, 0, 30, 40)
        >>> with open(gpath, 'rb') as img_file:
        ...     img = exif.Image(img_file)
        >>> img.gps_latitude = (1, 17, 30.786)
        >>> img.gps_latitude_ref = 'S'
        >>> img.gps_longitude = (36, 53, 53.4762)
        >>> img.gps_longitude_ref = 'W'
        >>> img.datetime = "2024:06:28 17:58:16"
        >>> img.orientation = 1
        >>> with open(gpath, 'wb') as img_file:
        ...     bytes = img_file.write(img.get_file())
        >>> pil_img = Image.open(gpath, 'r')
        >>> get_exif(pil_img, gpath)
        (1719611896, -1.291885, -36.89818783333333, 1, 30, 40)
        >>> shutil.rmtree(db_path + "test_dataset")
    """

    exif_dict = parse_exif(pil_img)
    unixtime = get_unixtime(exif_dict)
    lat, lon = get_lat_lon(exif_dict)
    orient = get_orientation(exif_dict)
    height, width = get_dim(exif_dict, gpath)
    return unixtime, lat, lon, orient, height, width


def get_standard_ext(gpath):
    """
    Returns standardized image extension.

    Parameters:
        gpath (str): image path

    Returns:
        ext (str): image file extension

    Doctest Command:
        python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE algo/preproc.py

    Example:
        >>> db_path = "doctest_files/"
        >>> gpath = db_path + 'test_dataset/images/img0.JPG'
        >>> get_standard_ext(gpath)
        '.jpg'
        >>> gpath = db_path + 'test_dataset/images/img0.jpeg'
        >>> get_standard_ext(gpath)
        '.jpg'
    """

    ext = splitext(gpath)[1].lower()
    return ".jpg" if ext == ".jpeg" else ext


def get_file_uuid(fpath, hasher=None, blocksize=65536, stride=1):
    """
    Creates a uuid from the hash of a file.

    Parameters:
        fpath (str): path to file
        hasher (hashlib hasher): hasher to generate file hash
        blocksize (int): size of block to read file with
        stride (int): used for skipping blocks

    Returns:
        uuid_ (str): deterministic uuid for input file

    Doctest Command:
        python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE algo/preproc.py

    Example:
        >>> import os
        >>> import numpy
        >>> import shutil
        >>> from PIL import Image
        >>> db_path = "doctest_files/"
        >>> os.makedirs(db_path + "test_data")
        >>> for n in range(2):
        ...     a = numpy.random.rand(30,30,3) * 255
        ...     img = Image.fromarray(a.astype('uint8')).convert('RGB')
        ...     img.save(db_path + ("test_data/img%000d.jpg" % n))
        >>> gpath_list = [db_path + "test_data/img0.jpg", db_path + "test_data/img0.jpg", db_path + "test_data/img1.jpg"]
        >>> uuid1 = get_file_uuid(gpath_list[0])
        >>> uuid2 = get_file_uuid(gpath_list[1])
        >>> uuid3 = get_file_uuid(gpath_list[2])
        >>> print(uuid1 == uuid2)
        True
        >>> print(uuid1 != uuid3)
        True
        >>> shutil.rmtree(db_path + "test_data")
    """
    hasher = hashlib.sha1()  # 20 bytes of output
    # sha1 produces a 20 byte hash
    with open(fpath, "rb") as file_:
        buf = file_.read(blocksize)
        while len(buf) > 0:
            hasher.update(buf)
            if stride > 1:
                file_.seek(blocksize * (stride - 1), 1)  # skip blocks
            buf = file_.read(blocksize)

        hashbytes_20 = hasher.digest()
    # sha1 produces 20 bytes, but UUID requires 16 bytes
    hashbytes_16 = hashbytes_20[0:16]
    uuid_ = uuid.UUID(bytes=hashbytes_16)
    return uuid_


def parse_imageinfo(gpath):
    """
    Gathers image exif data, deterministically calculates image uuid, converts image to 8-bit RGB, and normalizes orientation

    Parameters:
        gpath (str): image path (must be in UNIX-PATH format)

    Returns:
        param_tup (tuple): tuple of image parameters used to populate image table columns

    Doctest Command:
        python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE algo/preproc.py

    Example:
        >>> import os
        >>> import numpy
        >>> import exif
        >>> import shutil
        >>> from numpy.random import RandomState
        >>> from PIL import Image
        >>> db_path = "doctest_files/"
        >>> gpath = db_path + 'test_dataset/images/img0.JPG'
        >>> os.makedirs(db_path + "test_dataset/images")
        >>> prng = RandomState(0)
        >>> a = prng.rand(30, 30, 3) * 255
        >>> img = Image.fromarray(a.astype('uint8')).convert('RGB')
        >>> img.save(gpath)
        >>> parse_imageinfo(gpath)
        ('df53d013-889f-e6bf-2636-764a0cd2ce72',
        'doctest_files/test_dataset/images/img0.JPG',
        'doctest_files/test_dataset/images/img0.JPG',
        'img0.JPG', '.jpg', 30, 30, -1, -1, -1, 0, '')
        >>> with open(gpath, 'rb') as img_file:
        ...     img = exif.Image(img_file)
        >>> img.gps_latitude = (1, 17, 30.786)
        >>> img.gps_latitude_ref = 'N'
        >>> img.gps_longitude = (36, 53, 53.4762)
        >>> img.gps_longitude_ref = 'E'
        >>> img.datetime = "2024:06:28 17:58:16"
        >>> img.orientation = 1
        >>> with open(gpath, 'wb') as img_file:
        ...     bytes = img_file.write(img.get_file())
        >>> parse_imageinfo(gpath)
        ('f13d38dc-94da-15f9-1a69-99934d69e04a',
        'doctest_files/test_dataset/images/img0.JPG',
        'doctest_files/test_dataset/images/img0.JPG',
        'img0.JPG', '.jpg', 30, 30, 1719611896, 1.291885, 36.89818783333333, 1, '')
        >>> shutil.rmtree(db_path + "test_dataset")
    """

    # Try to open the image
    if gpath is None:
        return None, None
    elif isinstance(gpath, dict) and len(gpath) == 0:
        return None, None
    else:
        pass

    gpath = gpath.strip()

    try:
        # Open image with EXIF support to get time, GPS, and the original orientation
        pil_img = Image.open(gpath, "r")

        # Convert 16-bit RGBA images on disk to 8-bit RGB
        if pil_img.mode == "RGBA":
            pil_img.load()
            canvas = Image.new("RGB", pil_img.size, (255, 255, 255))
            canvas.paste(pil_img, mask=pil_img.split()[3])  # 3 is the alpha channel
            canvas.save(gpath)
            pil_img.close()

            # Reload image
            pil_img = Image.open(gpath, "r")

        img_time, lat, lon, orient, height, width = get_exif(
            pil_img, gpath
        )  # Read exif tags
        pil_img.close()

        if orient not in [EXIF_UNDEFINED, EXIF_NORMAL]:
            # OpenCV >= 3.1 supports EXIF tags, which will load correctly
            img = cv2.imread(gpath)
            assert img is not None

            try:
                # Sanitize weird behavior and standardize EXIF orientation to 1
                cv2.imwrite(gpath, img)
                orient = EXIF_NORMAL
            except AssertionError:
                return None, None
    except FileNotFoundError:
        return None, None

    # OpenCV imread too slow
    # Parse out the data
    # height, width = img.shape[:2]  # Read width, height

    # We cannot use pixel data as libjpeg is not deterministic (even for reads!)
    image_uuid = str(get_file_uuid(gpath))  # Read file ]-hash-> guid = gid

    orig_gname = basename(gpath)
    ext = get_standard_ext(gpath)
    notes = ""
    # Build parameters tuple
    param_tup = (
        image_uuid,
        gpath,
        gpath,
        orig_gname,
        ext,
        width,
        height,
        img_time,
        lat,
        lon,
        orient,
        notes,
    )

    return param_tup
