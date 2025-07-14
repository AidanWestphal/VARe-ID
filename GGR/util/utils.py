import cv2
import os
import shutil
import six
import numpy as np
from os.path import exists, expanduser, join, normpath, realpath, abspath, dirname
from PIL import Image, ExifTags

from GGR.util.constants import (
    ORIENTATION_DICT,
    ORIENTATION_000,
    ORIENTATION_090,
    ORIENTATION_180,
    ORIENTATION_270,
)

def path_from_file(file, relative):
    """
    Generates a correct path to a file from a relative path, regardless of the directory 
    the script is executed from. This is done by building from the absolute filepath of the 
    script itself.

    Parameters:
        file (str): The __file__ string of a file.
        relative (str): A relative filepath.
    
    Returns:
        abs_path (str): The absolute path found by following the relative filepath from the file path.
    """

    return os.path.join(dirname(abspath(file)), relative)


def parse_exif(pil_img):
    """
    Compiles image exif data into dictionary with string keys.

    Parameters:
        pil_img (open PIL image): image to extract exif data from

    Returns:
        exif_dict (dict): dictionary containing exif data with exif tag names as keys

    Doctest Command:
        python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE utils.py

    Example:
        >>> import os
        >>> import numpy
        >>> import exif
        >>> import shutil
        >>> from PIL import Image
        >>> db_path = "doctest_data/"
        >>> gpath = db_path + 'test_dataset/images/img0.JPG'
        >>> os.makedirs(db_path + "test_dataset/images")
        >>> a = numpy.random.rand(30,40,3) * 255
        >>> img = Image.fromarray(a.astype('uint8')).convert('RGB')
        >>> img.save(gpath)
        >>> pil_img = Image.open(gpath, 'r')
        >>> parse_exif(pil_img)
        {}
        >>> with open(gpath, 'rb') as img_file:
        ...     img = exif.Image(img_file)
        >>> img.gps_latitude = (1, 17, 30.786)
        >>> img.gps_latitude_ref = 'S'
        >>> img.gps_longitude = (36, 53, 53.4762)
        >>> img.gps_longitude_ref = 'E'
        >>> img.datetime = "2024:06:28 17:58:16"
        >>> img.orientation = 1
        >>> with open(gpath, 'wb') as img_file:
        ...     bytes = img_file.write(img.get_file())
        >>> pil_img = Image.open(gpath, 'r')
        >>> parse_exif(pil_img)
        {'DateTime': '2024:06:28 17:58:16', 'Orientation': 1, 'GPSLatitudeRef': 'S',
         'GPSLatitude': (1.0, 17.0, 30.786), 'GPSLongitudeRef': 'E', 'GPSLongitude': (36.0, 53.0, 53.4762)}
        >>> shutil.rmtree(db_path + "test_dataset")
    """

    img_exif = pil_img.getexif()
    exif_dict = dict()
    IFD_CODE_LOOKUP = {i.value: i.name for i in ExifTags.IFD}

    # Search for individual and grouped exif data
    for tag_code, value in img_exif.items():
        # Use tag names instead of tag codes
        if tag_code in IFD_CODE_LOOKUP:
            ifd_tag_name = IFD_CODE_LOOKUP[tag_code]
            ifd_data = img_exif.get_ifd(tag_code).items()

            for nested_key, nested_value in ifd_data:
                nested_tag_name = (
                    ExifTags.GPSTAGS.get(nested_key, None)
                    or ExifTags.TAGS.get(nested_key, None)
                    or nested_key
                )
                exif_dict[nested_tag_name] = nested_value
        else:
            exif_dict[ExifTags.TAGS.get(tag_code)] = value

    return exif_dict


def fix_orientation(pil_img, orient):
    """
    Ensures image is oriented right side up.

    Parameters:
        pil_img (open PIL image): image to fix
        orient (int): orientation of image

    Returns:
        pil_img (open PIL image): the updated image

    Doctest Command:
        python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE utils.py

    Example:
        >>> import os
        >>> import numpy
        >>> import exif
        >>> import shutil
        >>> from PIL import Image
        >>> db_path = "doctest_data/"
        >>> gpath = db_path + 'test_dataset/images/img0.JPG'
        >>> if os.path.exists(db_path + "test_dataset/images"):
        ...     shutil.rmtree(db_path + "test_dataset/images")
        >>> os.makedirs(db_path + "test_dataset/images")
        >>> a = numpy.random.rand(30,40,3) * 255
        >>> img = Image.fromarray(a.astype('uint8')).convert('RGB')
        >>> img.save(gpath)
        >>> with open(gpath, 'rb') as img_file:
        ...     img = exif.Image(img_file)
        >>> img.orientation = 6
        >>> with open(gpath, 'wb') as img_file:
        ...     bytes = img_file.write(img.get_file())
        >>> pil_img = Image.open(gpath)
        >>> img = fix_orientation(pil_img, 6)
        >>> print(type(img) == Image.Image)
        True
        >>> shutil.rmtree(db_path + "test_dataset")
    """

    assert orient in ORIENTATION_DICT
    orient_ = ORIENTATION_DICT[orient]
    if orient_ == ORIENTATION_000:
        return pil_img
    elif orient_ == ORIENTATION_090:
        return pil_img.rotate(90, expand=1)
    elif orient_ == ORIENTATION_180:
        return pil_img.rotate(180, expand=1)
    elif orient_ == ORIENTATION_270:
        return pil_img.rotate(270, expand=1)
    else:
        return pil_img


def fix_pil_img(pil_img):
    """
    Normalizes image color and orientation.

    Parameters:
        pil_img (open PIL image): image to be fixed

    Returns:
        imgBGR (ndarray): fixed image

    Doctest Command:
        python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE utils.py

    Example:
        >>> import os
        >>> import numpy
        >>> import exif
        >>> import shutil
        >>> from PIL import Image
        >>> db_path = "doctest_data/"
        >>> gpath = db_path + 'test_dataset/images/img0.JPG'
        >>> os.makedirs(db_path + "test_dataset/images")
        >>> a = numpy.random.rand(30,40,3) * 255
        >>> img = Image.fromarray(a.astype('uint8')).convert('RGB')
        >>> img.save(gpath)
        >>> with open(gpath, 'rb') as img_file:
        ...     img = exif.Image(img_file)
        >>> img.orientation = 1
        >>> with open(gpath, 'wb') as img_file:
        ...     bytes = img_file.write(img.get_file())
        >>> pil_img = Image.open(gpath)
        >>> img = fix_pil_img(pil_img)
        >>> print(type(img) == numpy.ndarray)
        True
        >>> shutil.rmtree(db_path + "test_dataset")
    """

    exif_dict = parse_exif(pil_img)
    orient = 0
    if "Orientation" in exif_dict.keys():
        orient = exif_dict["Orientation"]

    if orient in ORIENTATION_DICT:
        pil_img_fixed = fix_orientation(pil_img, orient)
    np_img = np.array(pil_img_fixed.convert("RGB"))
    imgBGR = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    if not isinstance(orient, bool) and orient in ORIENTATION_DICT:
        imgBGR = fix_orientation(imgBGR, orient)
    return imgBGR


def duplicates_exist(items):
    """returns if list has duplicates"""
    return len(items) - len(set(items)) != 0


def imread(img_fpath):
    """
    Wrapper around the opencv imread function. Handles remote uris.

    Parameters:
        img_fpath (str):  file path string
        grayscale (bool): (default = False)
        orient (bool): (default = False)

    Returns:
        ndarray: imgBGR

    Doctest Command:
        python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE utils.py

    Example:
        >>> import os
        >>> import numpy
        >>> import exif
        >>> import shutil
        >>> from PIL import Image
        >>> db_path = "doctest_data/"
        >>> gpath = db_path + 'test_dataset/images/img0.JPG'
        >>> os.makedirs(db_path + "test_dataset/images")
        >>> a = numpy.random.rand(30,40,3) * 255
        >>> img = Image.fromarray(a.astype('uint8')).convert('RGB')
        >>> img.save(gpath)
        >>> with open(gpath, 'rb') as img_file:
        ...     img = exif.Image(img_file)
        >>> img.orientation = 1
        >>> with open(gpath, 'wb') as img_file:
        ...     bytes = img_file.write(img.get_file())
        >>> img = imread(gpath)
        >>> print(type(img) == numpy.ndarray)
        True
        >>> shutil.rmtree(db_path + "test_dataset")
    """
    path, ext = os.path.splitext(img_fpath)
    try:
        with Image.open(img_fpath) as pil_img:
            imgBGR = fix_pil_img(pil_img)
    except Exception as ex:
        imgBGR = None
    if imgBGR is None:
        if not exists(img_fpath):
            raise IOError("cannot read img_fpath=%s does not exist." % img_fpath)
        else:
            if not os.access(img_fpath, os.R_OK):
                raise PermissionError(
                    "cannot read img_fpath={} access denied.".format(img_fpath)
                )

            msg = (
                "Cannot read img_fpath=%s, "
                "seems corrupted or memory error." % img_fpath
            )
            print("[utils.imread] " + msg)
            raise IOError(msg)
    return imgBGR


def isiterable(obj):
    """
    Returns if the object can be iterated over and is NOT a string.

    Parameters:
        obj (scalar or iterable): object to test

    Returns:
        (bool): whether the object is iterable

    Doctest Command:
        python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE utils.py

    Example:
        >>> # build test data
        >>> obj_list = [3, [3], '3', (3,), [3,4,5]]
        >>> # execute function
        >>> result = [isiterable(obj) for obj in obj_list]
        >>> # verify results
        >>> print(result)
        [False, True, False, True, True]
    """

    try:
        iter(obj)
        return not isinstance(obj, six.string_types)
    except Exception:
        return False


def list_compress(item_list, flag_list, inverse=False):
    """
    Returns items in item list where the corresponding item in flag list is
    True (False if inverse is True).

    Parameters:
        item_list (list): list of items to mask
        flag_list (list): list of booleans used as a mask
        inverse (bool): boolean used to determine whether to filter True or False items

    Returns:
        filtered_items (list): filtered item list

    Doctest Command:
        python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE utils.py

    Example:
        >>> list_compress([1, 2, 3, 4, 5], [True, False, True, False, True])
        [1, 3, 5]
        >>> list_compress([1, 2, 3, 4, 5], [True, False, True, False, True], inverse=True)
        [2, 4]
    """

    assert len(item_list) == len(
        flag_list
    ), "lists should correspond. len(item_list)=%r len(flag_list)=%r" % (
        len(item_list),
        len(flag_list),
    )

    filtered_items = []
    for item_idx in range(len(item_list)):
        if (not inverse and flag_list[item_idx]) or (
            inverse and not flag_list[item_idx]
        ):
            filtered_items.append(item_list[item_idx])

    return filtered_items


def list_unique(item_list):
    """
    Returns item list without any duplicates.
    Maintains item order.

    Parameters:
        item_list (list): list of items to screen

    Returns:
        (list): filtered item list

    Doctest Command:
        python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE utils.py

    Example:
        >>> list_unique([3, 3, 3, 4, 2, 1, 1, 4, 6])
        [3, 4, 2, 1, 6]
    """

    flag_list = []
    for item_idx in range(len(item_list)):
        (
            flag_list.append(False)
            if item_list[item_idx] in item_list[:item_idx]
            else flag_list.append(True)
        )

    return list_compress(item_list, flag_list)


def copy_file_list(src_list, dst_list, err_ok=False):
    """
    Copies series of files and preserves metadata.

    Parameters:
        src_list (list): list of file paths to copy
        dst_list (list): list of file paths to copy into
        err_ok (bool): if true, returns false for any failed copies instead of producing an error

    Returns:
        success_list (list): list of flags (true for successful copies, false for failed copies)

    Doctest Command:
        python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE utils.py

    Example:
        >>> path = ''
        >>> src_list = [path + 'src_dir/img1.jpg', path + 'src_dir/img2.jpg']
        >>> dst_list = [path + 'dst_dir/img1.jpg', path + 'dst_dir/img2.jpg']
        >>> os.mkdir(path + 'src_dir/')
        >>> os.mkdir(path + 'dst_dir/')
        >>> img = Image.new('RGB',(480,640),'rgb(255,255,255)')
        >>> img.save(src_list[0])
        >>> img = Image.new('RGB',(480,640),'rgb(0,0,0)')
        >>> img.save(src_list[1])
        >>> copy_file_list(src_list, dst_list)
        ['dst_dir/img1.jpg', 'dst_dir/img2.jpg']
        >>> shutil.rmtree(path + 'src_dir/')
        >>> shutil.rmtree(path + 'dst_dir/')
    """

    success_list = []
    for src, dst in zip(src_list, dst_list):
        try:
            path = shutil.copy2(src, dst)
            success_list.append(path)
        except:
            if err_ok:
                success_list.append(None)
            else:
                raise

    return success_list


def ensuredir(path_, mode=0o1777):
    """
    Ensures that directory will exist. creates new dir with sticky bits by
    default.

    Parameters:
        path_ (str): path to ensure. Can also be a tuple to send to join
        mode (int): octal mode of directory (default 0o1777)

    Returns:
        path_ (str): path of the ensured directory

    Doctest Command:
        python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE utils.py

    Example:
        >>> path = 'test_dir/'
        >>> ensuredir(path)
        [ensuredir] mkdir('test_dir/')
        'test_dir/'
        >>> os.path.exists(path)
        True
        >>> os.rmdir(path)
    """

    if isinstance(path_, (list, tuple)):
        path_ = join(*path_)

    if not os.path.exists(path_):
        print("[ensuredir] mkdir(%r)" % path_)
        os.makedirs(normpath(path_), mode=mode)

    return path_


def remove_file(fpath, ignore_errors=True):
    """
    Removes a file.

    Parameters:
        fpath (str): file path to remove
        ignore_errors (bool): if true, ignores errors

    Returns:
        (bool): whether or not removal was a success

    Doctest Command:
        python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE utils.py

    Example:
        >>> with open('test_file.txt', 'w') as file:
        ...     file.write('test')
        4
        >>> remove_file(os.path.abspath('test_file.txt')) # doctest: +ELLIPSIS
        [utils.remove_file] Finished deleting path='...test_file.txt'
        True
    """

    try:
        os.remove(fpath)
    except OSError:
        print("[utils.remove_file] Could not delete %s" % fpath)
        if not ignore_errors:
            raise

        return False

    print("[utils.remove_file] Finished deleting path=%r" % fpath)
    return True


def unixpath(path):
    """
    Corrects fundamental problems with windows paths.

    Parameters:
        path (str): path to correct

    Returns:
        path (str): corrected path

    Doctest Command:
        python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE utils.py

    Example:
        >>> unixpath('/mnt/c/Users\\Julian\\Images\\img.jpg')
        '/mnt/c/Users/Julian/Images/img.jpg'
    """

    return normpath(realpath(expanduser(path))).replace("\\", "/")


def ensure_unix_gpaths(gpath_list):
    """
    Asserts that all paths are given with forward slashes.
    If not it fixes them.

    Parameters:
        gpath_list (list): list of image paths

    Returns:
        gpath_list (list): list of updated image paths

    Doctest Command:
        python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE utils.py

    Example:
        >>> ensure_unix_gpaths(['/mnt/c/Users/Julian/Images/img1.jpg', '/mnt/c/Users/Julian/Images\\img2.jpg'])
        ['/mnt/c/Users/Julian/Images/img1.jpg', '/mnt/c/Users/Julian/Images/img2.jpg']
    """

    gpath_list_ = []
    for count, gpath in enumerate(gpath_list):
        if gpath is None:
            gpath = None
        elif isinstance(gpath, dict) and len(gpath) == 0:
            gpath = None
        else:
            try:
                msg = (
                    "gpath_list must be in unix format (no backslashes)."
                    "Failed on %d-th gpath=%r"
                )
                assert gpath.find("\\") == -1, msg % (count, gpath)
            except (AttributeError, AssertionError):
                gpath = unixpath(gpath)
        gpath_list_.append(gpath)

    return gpath_list_
