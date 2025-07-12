import cv2
from os.path import exists, isabs, join
from pathlib import Path
from PIL import Image

import GGR.util.constants as const
import GGR.util.utils as ut
from GGR.util import preproc
from GGR.util.ggr_funcs import extrapolate_ggr_gps
import GGR.util.ggr_funcs as ggr

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


def _compute_image_params(gpath_list, sanitize=True, ensure=True, doctest_mode=False):
    """
    Compute image uuids and collect image exif data

    Parameters:
        gpath_list (list): list of image paths
        sanitize (bool): if true, fixes image paths
        ensure (bool): if true, reports additional data on import failure
        doctest_mode (bool): if true, replaces carriage returns with newlines

    Returns:
        params_list (list): list of tuples containing image metadata

    Doctest Command:
        python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE control/image_funcs.py

    Example:
        >>> import os
        >>> import numpy
        >>> import shutil
        >>> from PIL import Image
        >>> db_path = "doctest_files/"
        >>> if os.path.exists(db_path):
        ...     shutil.rmtree(db_path)
        >>> os.makedirs(db_path + "test_data")
        >>> for n in range(2):
        ...     a = numpy.random.rand(30,30,3) * 255
        ...     img = Image.fromarray(a.astype('uint8')).convert('RGB')
        ...     img.save(db_path + ("test_data/img%000d.jpg" % n))
        >>> gpath_list = [db_path + "test_data/img0.jpg", db_path + "test_data/img0.jpg", db_path + "test_data/img1.jpg"]
        >>> uuid_list = _compute_image_params(gpath_list, doctest_mode=True)
        [parse_imageinfo] parsing images [1/3]
        [parse_imageinfo] parsing images [2/3]
        [parse_imageinfo] parsing images [3/3]
        >>> print(uuid_list[0][0] == uuid_list[1][0])
        True
        >>> print(uuid_list[0][0] != uuid_list[2][0])
        True
        >>> shutil.rmtree(db_path + "test_data")
    """

    if not gpath_list:
        return []

    # Processing an image might fail, yeilding a None instead of a tup
    if sanitize:
        gpath_list = ut.ensure_unix_gpaths(gpath_list)

    # Create param_iter
    params_list = []
    for gpath_idx in range(len(gpath_list)):
        params_list.append(preproc.parse_imageinfo(gpath_list[gpath_idx]))
        if doctest_mode:
            print(
                "[parse_imageinfo] parsing images [%d/%d]\n"
                % (gpath_idx + 1, len(gpath_list)),
                end="",
            )
        else:
            print(
                "[parse_imageinfo] parsing images [%d/%d]\r"
                % (gpath_idx + 1, len(gpath_list)),
                end="",
            )

    # Error reporting
    failed_list = [
        gpath for (gpath, params_) in zip(gpath_list, params_list) if not params_
    ]

    print(
        "\n".join(
            [" ! Failed reading gpath={!r}".format(gpath) for gpath in failed_list]
        )
    )

    if ensure and len(failed_list) > 0:
        print(f"Importing {len(failed_list)} files failed: {failed_list}")

    return params_list


def compute_image_uuids(gpath_list, doctest_mode=False, **kwargs):
    """
    Compute image uuids

    Parameters:
        gpath_list (list): list of image paths
        doctest_mode (bool): if true, replaces carriage returns with newlines

    Returns:
        params_list (list): list of tuples containing image metadata

    Doctest Command:
        python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE control/image_funcs.py

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
        >>> uuid_list = compute_image_uuids(gpath_list, doctest_mode=True)
        [parse_imageinfo] parsing images [1/3]
        [parse_imageinfo] parsing images [2/3]
        [parse_imageinfo] parsing images [3/3]
        >>> print(uuid_list[0] == uuid_list[1])
        True
        >>> print(uuid_list[0] != uuid_list[2])
        True
        >>> shutil.rmtree(db_path + "test_data")
    """
    params_list = _compute_image_params(gpath_list, doctest_mode, **kwargs)

    uuid_colx = IMAGE_COLNAMES.index("uuid")
    uuid_list = [
        None if params_ is None else params_[uuid_colx] for params_ in params_list
    ]

    return uuid_list


def filter_image_set(
    imgtable,
    gid_list,
    require_unixtime=False,
    require_gps=False,
    is_reviewed=False,
    sort=False,
):
    """
    Filters undesired images out of image set

    Parameters:
        imgtable (ImageTable): table containing image metadata
        gid_list (list): list of image gids
        require_unixtime (bool): if true, filters out images that do not have time data
        require_gps (bool): if true, filters our images that do not have gps data
        is_reviewed (bool): if true, filters out images that are not reviewed
        sort (bool): if true, sorts image set

    Returns:
        gid_list (list): filtered list of image gids

    Doctest Command:
        python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE control/image_funcs.py

    Example:
        >>> from db.table import ImageTable
        >>> table = ImageTable("doctest_files/", ["grevy's zebra"])
        >>> gid_list = table.add_image_data(['uuid', 'time_posix', 'gps_lat', 'gps_lon', 'reviewed'],
        ... [['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', 1579961262, 0.291885, 36.89818833333333, True],
        ... ['11ca4e9b-1de9-4ee5-8d90-615bf93077c8', -1, 0.291885, 36.89818833333333, True],
        ... ['6830cd43-ca00-28ee-8843-fe64d789d7f7', 1579961262, -1, -1, True],
        ... ['b54193f2-8507-8441-5bf3-cd9f0ed8a883', 1579961262, 0.291885, 36.89818833333333, False]])
        >>> filter_image_set(table, gid_list)
        [1, 2, 3, 4]
        >>> filter_image_set(table, gid_list, require_unixtime=True)
        [1, 3, 4]
        >>> filter_image_set(table, gid_list, require_gps=True)
        [1, 2, 4]
        >>> filter_image_set(table, gid_list, is_reviewed=True)
        [1, 2, 3]
        >>> filter_image_set(table, gid_list, require_unixtime=True, require_gps=True, is_reviewed=True)
        [1]
    """

    if require_unixtime:
        # Remove images without timestamps
        unixtime_list = imgtable.get_image_unixtimes(gid_list)
        isvalid_list = [unixtime != -1 for unixtime in unixtime_list]
        gid_list = ut.list_compress(gid_list, isvalid_list)
    if require_gps:
        # Remove images without gps
        isvalid_gps = [
            lat != -1 and lon != -1 for lat, lon in imgtable.get_image_gps(gid_list)
        ]
        gid_list = ut.list_compress(gid_list, isvalid_gps)

    if is_reviewed:
        reviewed_list = imgtable.get_image_reviewed(gid_list)
        isvalid_list = [is_reviewed == flag for flag in reviewed_list]
        gid_list = ut.list_compress(gid_list, isvalid_list)

    if sort:
        gid_list = sorted(gid_list)
    return gid_list


def get_valid_gids(
    imgtable,
    imgsetid=None,
    imgsetid_list=[],
    require_unixtime=False,
    require_gps=False,
    is_reviewed=False,
    sort=False,
):
    """
    Gets all valid gids from images stored in imgtable

    Parameters:
        imgtable (ImageTable): table containing image metadata
        imgsetid (int): if not none, only gets valid gids from images in specified image sets
        imgsetid_list (list): list of image set ids to get gids from
        require_unixtime (bool): if true, filters out images that do not have time data
        require_gps (bool): if true, filters our images that do not have gps data
        is_reviewed (bool): if true, filters out images that are not reviewed
        sort (bool): if true, sorts image set

    Returns:
        list: gid_list

    Doctest Command:
        python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE control/image_funcs.py

    Example:
        >>> from db.table import ImageTable
        >>> table = ImageTable("doctest_files/", ["grevy's zebra"])
        >>> gid_list = table.add_image_data(['uuid', 'time_posix', 'gps_lat', 'gps_lon', 'reviewed'],
        ... [['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', 1579961262, 0.291885, 36.89818833333333, True],
        ... ['11ca4e9b-1de9-4ee5-8d90-615bf93077c8', -1, 0.291885, 36.89818833333333, True],
        ... ['6830cd43-ca00-28ee-8843-fe64d789d7f7', 1579961262, -1, -1, True],
        ... ['b54193f2-8507-8441-5bf3-cd9f0ed8a883', 1579961262, 0.291885, 36.89818833333333, False]])
        >>> get_valid_gids(table) == gid_list
        True
        >>> get_valid_gids(table, require_unixtime=True)
        [1, 3, 4]
        >>> get_valid_gids(table, require_gps=True)
        [1, 2, 4]
        >>> get_valid_gids(table, is_reviewed=True)
        [1, 2, 3]
        >>> get_valid_gids(table, require_unixtime=True, require_gps=True, is_reviewed=True)
        [1]
    """

    if imgsetid is None and not imgsetid_list:
        gid_list = imgtable.get_all_gids()
    elif imgsetid_list:
        gid_list = imgtable.get_imgset_gids(imgsetid_list)
    else:
        assert not ut.isiterable(imgsetid)
        gid_list = imgtable.get_imgset_gids(imgsetid)

    gid_list = filter_image_set(
        imgtable,
        gid_list,
        require_unixtime=require_unixtime,
        require_gps=require_gps,
        is_reviewed=is_reviewed,
        sort=sort,
    )

    return gid_list


def localize_images(imgtable, gid_list_=None):
    """
    Moves the images into the designated image cache.
    Images are renamed to img_uuid.ext.

    Parameters:
        imgtable (ImageTable): table containing image metadata
        gid_list_ (list): list of image gids

    Doctest Command:
        python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE control/image_funcs.py

    Example:
        >>> import os
        >>> import numpy
        >>> import shutil
        >>> from PIL import Image
        >>> from db.table import ImageTable
        >>> db_path = "doctest_files/"
        >>> os.makedirs(db_path + "test_data/QR_100/Day1")
        >>> os.makedirs(db_path + "test_data/QR_100/Day2")
        >>> os.makedirs(db_path + "test_dataset/images")
        >>> for n in range(2):
        ...     a = numpy.random.rand(30,30,3) * 255
        ...     img = Image.fromarray(a.astype('uint8')).convert('RGB')
        ...     img.save(db_path + ('test_data/QR_100/Day1/img%000d.jpg' % n))
        >>> for n in range(2):
        ...     a = numpy.random.rand(30,30,3) * 255
        ...     img = Image.fromarray(a.astype('uint8')).convert('RGB')
        ...     img.save(db_path + ('test_data/QR_100/Day2/img%000d.jpg' % n))
        >>> table = ImageTable(db_path + "test_dataset/", ["grevy's zebra"])
        >>> gid_list = table.add_image_data(['uuid', 'uri', 'original_name', 'ext'],
        ... [['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', db_path + 'test_data/QR_100/Day1/img0.JPG', 'img0', '.JPG'],
        ... ['11ca4e9b-1de9-4ee5-8d90-615bf93077c8', db_path + 'test_data/QR_100/Day1/img1.JPG', 'img1', '.JPG'],
        ... ['6830cd43-ca00-28ee-8843-fe64d789d7f7', db_path + 'test_data/QR_100/Day2/img0.JPG', 'img0', '.JPG'],
        ... ['b54193f2-8507-8441-5bf3-cd9f0ed8a883', db_path + 'test_data/QR_100/Day2/img1.JPG', 'img1', '.JPG']])
        >>> localize_images(table, gid_list)
        Localizing doctest_files/test_data/QR_100/Day1/img0.JPG ->
        doctest_files/test_dataset/images/0fdef8e8-cec0-b460-bac2-6ee3e39f0798.JPG
            ...image copied
        Localizing doctest_files/test_data/QR_100/Day1/img1.JPG ->
        doctest_files/test_dataset/images/11ca4e9b-1de9-4ee5-8d90-615bf93077c8.JPG
            ...image copied
        Localizing doctest_files/test_data/QR_100/Day2/img0.JPG ->
        doctest_files/test_dataset/images/6830cd43-ca00-28ee-8843-fe64d789d7f7.JPG
            ...image copied
        Localizing doctest_files/test_data/QR_100/Day2/img1.JPG ->
        doctest_files/test_dataset/images/b54193f2-8507-8441-5bf3-cd9f0ed8a883.JPG
            ...image copied
        >>> print(len([name for name in os.listdir(db_path + "test_dataset/images")
        ...            if os.path.isfile(os.path.join(db_path + "test_dataset/images", name)) and name[-3:] == "JPG"]))
        4
        >>> print(table.get_image_uris(gid_list))
        ['0fdef8e8-cec0-b460-bac2-6ee3e39f0798.JPG', '11ca4e9b-1de9-4ee5-8d90-615bf93077c8.JPG',
         '6830cd43-ca00-28ee-8843-fe64d789d7f7.JPG', 'b54193f2-8507-8441-5bf3-cd9f0ed8a883.JPG']
        >>> table.set_image_uris(gid_list, ['test_data/QR_100/Day1/img0.JPG', 'test_data/QR_100/Day1/img1.JPG',
        ...                                 'test_data/QR_100/Day2/img0.JPG', 'test_data/QR_100/Day2/img1.JPG'])
        >>> localize_images(table, gid_list)
        Localizing test_data/QR_100/Day1/img0.JPG -> doctest_files/test_dataset/images/0fdef8e8-cec0-b460-bac2-6ee3e39f0798.JPG
            ...skipping (already localized)
        Localizing test_data/QR_100/Day1/img1.JPG -> doctest_files/test_dataset/images/11ca4e9b-1de9-4ee5-8d90-615bf93077c8.JPG
            ...skipping (already localized)
        Localizing test_data/QR_100/Day2/img0.JPG -> doctest_files/test_dataset/images/6830cd43-ca00-28ee-8843-fe64d789d7f7.JPG
            ...skipping (already localized)
        Localizing test_data/QR_100/Day2/img1.JPG -> doctest_files/test_dataset/images/b54193f2-8507-8441-5bf3-cd9f0ed8a883.JPG
            ...skipping (already localized)
        >>> shutil.rmtree(db_path + "test_data")
        >>> shutil.rmtree(db_path + "test_dataset")
    """

    if gid_list_ is None:
        print("WARNING: you are localizing all gids")
        gid_list_ = get_valid_gids(imgtable)
    isvalid_list = [gid is not None for gid in gid_list_]
    gid_list = ut.list_unique(ut.list_compress(gid_list_, isvalid_list))
    # uri_list = imgtable.get_image_uris(gid_list)

    # def islocal(uri):
    #     return not isabs(uri)

    guuid_list = imgtable.get_image_uuids(gid_list)
    gext_list = imgtable.get_image_exts(gid_list)
    # Build list of image names based on uuid in the imgtable.imgdir
    guuid_strs = (str(guuid) for guuid in guuid_list)
    loc_gname_list = [guuid + ext for (guuid, ext) in zip(guuid_strs, gext_list)]
    # loc_gpath_list = [join(imgtable.imgdir, gname) for gname in loc_gname_list]
    # Copy images to local directory

    # for uri, loc_gpath in zip(uri_list, loc_gpath_list):
    #     print(f"Localizing {uri} -> {loc_gpath}")

    #     if not exists(loc_gpath):
    #         uri if islocal(uri) else join(imgtable.imgdir, uri)
    #         ut.copy_file_list([uri], [loc_gpath])
    #         print("\t...image copied")
    #     else:
    #         print("\t...skipping (already localized)")

    # Update database uris
    imgtable.set_image_uris(gid_list, loc_gname_list)
    # assert all(map(exists, loc_gpath_list)), "not all images copied"


def check_image_loadable_worker(gpath, orient):
    """
    Check whether an image can be loaded and standardize exif orientation.

    Parameters:
        gpath (str): path to image
        orient (int): orientation of image

    Returns:
        loadable (bool): whether or not the image is loadable
        rewritten (bool): whether or not the image was rewritten
        orient (bool): updated image orientation

    Doctest Command:
        python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE control/image_funcs.py

    Example:
        >>> import os
        >>> import numpy
        >>> import shutil
        >>> from PIL import Image
        >>> db_path = "doctest_files/"
        >>> os.makedirs(db_path + "test_dataset/images")
        >>> a = numpy.random.rand(30,30,3) * 255
        >>> img = Image.fromarray(a.astype('uint8')).convert('RGB')
        >>> img.save(db_path + 'test_dataset/images/img0.JPG')
        >>> check_image_loadable_worker(db_path + "test_dataset/images/img0.JPG", 0)
        (True, False, 0)
        >>> check_image_loadable_worker(db_path + "test_dataset/images/img0.JPG", 15)
        (True, True, 1)
        >>> with open(db_path + "test_dataset/images/fake_img.txt", "w") as img:
        ...     size = img.write("fake")
        >>> check_image_loadable_worker(db_path + "test_dataset/images/fake_img.txt", 0)
        [utils.imread] Cannot read img_fpath=doctest_files/test_dataset/images/fake_img.txt,
        seems corrupted or memory error.
        (False, False, 0)
        >>> shutil.rmtree(db_path + "test_dataset")
    """

    loadable, exif, rewritten, orient = True, True, False, orient
    try:
        if orient not in [EXIF_UNDEFINED, EXIF_NORMAL]:
            img = cv2.imread(gpath)
            assert img is not None
            # Sanitize weird behavior and standardize EXIF orientation to 1
            cv2.imwrite(gpath, img)
            orient = EXIF_NORMAL
            rewritten = True

        img = ut.imread(gpath)
        assert img is not None
    except Exception:
        loadable = False
    return loadable, rewritten, orient


# Check whether images are loadable
def check_image_loadable(imgtable, gid_list=None, doctest_mode=False):
    """
    Check whether images are loadable and standardize exif orientation.

    Parameters:
        imgtable (ImageTable): table containing image metadata
        gid_list (list): list of image gids
        doctest_mode (bool): if true, replaces carriage returns with newlines

    Returns:
        bad_loadable_list (list): list of gids of unloadable images

    Doctest Command:
        python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE control/image_funcs.py

    Example:
        >>> import os
        >>> import numpy
        >>> import shutil
        >>> from PIL import Image
        >>> from db.table import ImageTable
        >>> db_path = "doctest_files/"
        >>> os.makedirs(db_path + "test_dataset/images")
        >>> for n in range(4):
        ...     a = numpy.random.rand(30,30,3) * 255
        ...     img = Image.fromarray(a.astype('uint8')).convert('RGB')
        ...     img.save(db_path + ('test_dataset/images/img%000d.JPG' % n))
        >>> table = ImageTable(db_path + "test_dataset", ["grevy's zebra"])
        >>> gid_list = table.add_image_data(['uuid', 'uri', 'original_name', 'ext', 'orientation'],
        ... [['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', 'img0.JPG', 'img0', '.JPG', 0],
        ... ['11ca4e9b-1de9-4ee5-8d90-615bf93077c8', 'img1.JPG', 'img1', '.JPG', 0],
        ... ['6830cd43-ca00-28ee-8843-fe64d789d7f7', 'img2.JPG', 'img0', '.JPG', 0],
        ... ['b54193f2-8507-8441-5bf3-cd9f0ed8a883', 'img3.JPG', 'img1', '.JPG', 0]])
        >>> check_image_loadable(table, gid_list, doctest_mode=True)
        checking image loadable
        [check_image_loadable] validating images [1/4]
        [check_image_loadable] validating images [2/4]
        [check_image_loadable] validating images [3/4]
        [check_image_loadable] validating images [4/4]
        []
        >>> shutil.rmtree(db_path + "test_dataset")
    """

    print("checking image loadable")
    if gid_list is None:
        gid_list = get_valid_gids(imgtable)

    gpath_list = imgtable.get_image_paths(gid_list)
    existing_orient_list = imgtable.get_image_orientations(gid_list)

    loadable_list, rewritten_list, new_orient_list = [], [], []
    for gpath_idx in range(len(gpath_list)):
        gpath, orient = gpath_list[gpath_idx], existing_orient_list[gpath_idx]
        loadable, rewritten, new_orient = check_image_loadable_worker(gpath, orient)
        loadable_list.append(loadable), rewritten_list.append(
            rewritten
        ), new_orient_list.append(new_orient)
        if doctest_mode:
            print(
                "[check_image_loadable] validating images [%d/%d]\n"
                % (gpath_idx + 1, len(gpath_list)),
                end="",
            )
        else:
            print(
                "[check_image_loadable] validating images [%d/%d]\r"
                % (gpath_idx + 1, len(gpath_list)),
                end="",
            )

    print()
    update_gid_list = []
    update_orient_list = []
    zipped = list(zip(gid_list, existing_orient_list, new_orient_list))
    for gid, existing_orient, new_orient in zipped:
        if existing_orient != new_orient:
            if existing_orient == EXIF_UNDEFINED and new_orient == EXIF_NORMAL:
                # Update undefined to normal orient
                continue
            update_gid_list.append(gid)
            update_orient_list.append(new_orient)

    if len(update_gid_list) > 0:
        assert len(update_gid_list) == len(update_orient_list)
        args = (
            len(update_gid_list),
            len(rewritten_list),
        )
        print(
            f"[check_image_loadable] Updating {len(update_gid_list)} orientations from {len(rewritten_list)} rewritten images"
        )
        imgtable.set_image_orientation(update_gid_list, update_orient_list)

    bad_loadable_list = ut.list_compress(gid_list, loadable_list, inverse=True)
    return bad_loadable_list


def check_image_bit_depth_worker(gpath):
    """
    Check bit depth of image and convert to 8-bit RGB if necessary.

    Parameters:
        gpath (str): path to image

    Returns:
        flag (bool): None if image was not changed, true if image was successfully changed,
        false if image could not be changed

    Doctest Command:
        python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE control/image_funcs.py

    Example:
        >>> import os
        >>> import numpy
        >>> import shutil
        >>> from PIL import Image
        >>> db_path = "doctest_files/"
        >>> os.makedirs(db_path + "test_dataset/images")
        >>> a = numpy.random.rand(30,30,3) * 255
        >>> img = Image.fromarray(a.astype('uint8')).convert('RGB')
        >>> img.save(db_path + 'test_dataset/images/img0.JPG')
        >>> check_image_bit_depth_worker(db_path + "test_dataset/images/img0.JPG")
        >>> shutil.rmtree(db_path + "test_dataset")
    """

    flag = None
    try:
        img = Image.open(gpath, "r")
        assert img is not None

        # Convert 16-bit RGBA images on disk to 8-bit RGB
        if img.mode == "RGBA":
            img.load()

            canvas = Image.new("RGB", img.size, (255, 255, 255))
            canvas.paste(img, mask=img.split()[3])  # 3 is the alpha channel
            canvas.save(gpath)
            canvas = None
            flag = True

        img.close()
    except Exception:
        flag = False
    return flag


def check_image_bit_depth(imgtable, gid_list=None, doctest_mode=False):
    """
    Check bit depth of images and convert to 8-bit RGB when necessary.
    Also checks uuids for correctness.

    Parameters:
        imgtable (ImageTable): table containing image metadata
        gid_list (list): list of image gids
        doctest_mode (bool): if true, replaces carriage returns with newlines

    Returns:
        update_gid_list (list): list of gids of images that were updated
        update_uuid_list (list): list of updated uuids

    Doctest Command:
        python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE control/image_funcs.py

    Example:
        >>> import os
        >>> import numpy
        >>> import shutil
        >>> from PIL import Image
        >>> from db.table import ImageTable
        >>> db_path = "doctest_files/"
        >>> os.makedirs(db_path + "test_dataset/images")
        >>> for n in range(4):
        ...     a = numpy.random.rand(30,30,3) * 255
        ...     img = Image.fromarray(a.astype('uint8')).convert('RGB')
        ...     img.save(db_path + ('test_dataset/images/img%000d.JPG' % n))
        >>> table = ImageTable(db_path + "test_dataset", ["grevy's zebra"])
        >>> gid_list = table.add_image_data(['uuid', 'uri', 'original_name', 'ext', 'orientation'],
        ... [['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', 'img0.JPG', 'img0', '.JPG', 0],
        ... ['11ca4e9b-1de9-4ee5-8d90-615bf93077c8', 'img1.JPG', 'img1', '.JPG', 0],
        ... ['6830cd43-ca00-28ee-8843-fe64d789d7f7', 'img2.JPG', 'img0', '.JPG', 0],
        ... ['b54193f2-8507-8441-5bf3-cd9f0ed8a883', 'img3.JPG', 'img1', '.JPG', 0]])
        >>> check_image_bit_depth(table, gid_list, doctest_mode=True)
        checking image depth
        [check_image_bit_depth] checking image bit depth [1/4]
        [check_image_bit_depth] checking image bit depth [2/4]
        [check_image_bit_depth] checking image bit depth [3/4]
        [check_image_bit_depth] checking image bit depth [4/4]
        [check_image_bit_depth] updated 0 images
        ([], [])
        >>> shutil.rmtree(db_path + "test_dataset")
    """

    print("checking image depth")
    if gid_list is None:
        gid_list = get_valid_gids(imgtable)

    gpath_list = imgtable.get_image_paths(gid_list)
    flag_list = []

    for gpath_idx in range(len(gpath_list)):
        flag_list.append(check_image_bit_depth_worker(gpath_list[gpath_idx]))
        if doctest_mode:
            print(
                "[check_image_bit_depth] checking image bit depth [%d/%d]\n"
                % (gpath_idx + 1, len(gpath_list)),
                end="",
            )
        else:
            print(
                "[check_image_bit_depth] checking image bit depth [%d/%d]\r"
                % (gpath_idx + 1, len(gpath_list)),
                end="",
            )

    print()
    update_gid_list = ut.list_compress(gid_list, flag_list)

    print(f"[check_image_bit_depth] updated {len(update_gid_list)} images")

    update_gpath_list = imgtable.get_image_paths(update_gid_list)
    update_uuid_list = compute_image_uuids(update_gpath_list)

    return update_gid_list, update_uuid_list


def add_images(
    imgtable,
    gpath_list,
    params_list=None,
    as_annots=False,
    auto_localize=True,
    location_for_names="GGR",
    ensure_loadable=True,
    doctest_mode=False,
):
    """
    Adds a list of image paths to the image table.

    Initially we set the image_uri to exactly the given gpath.
    Later we change the uri, but keeping it the same here lets
    us process images asychronously.

    Parameters:
        imgtable (ImageTable): table to add image metadata to
        gpath_list (list): list of image paths to add
        params_list (list): metadata list for corresponding images that can either be
            specified outright or can be parsed from the image data directly if None
        as_annots (bool): if True, an annotation is automatically added for the entire
            image
        auto_localize (bool): if None uses the default specified in ibs.cfg
        location_for_names (str):
        ensure_loadable (bool): check whether imported images can be loaded.  Defaults to
            True
        doctest_mode (bool): if true, replaces carriage returns with newlines

    Returns:
        gid_list (list of rowids): gids are image rowids

    Doctest Command:
        python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE control/image_funcs.py

    Example:
        >>> import os
        >>> import numpy
        >>> import shutil
        >>> from numpy.random import RandomState
        >>> from PIL import Image
        >>> from db.table import ImageTable
        >>> db_path = "doctest_files/"
        >>> if os.path.exists(db_path):
        ...     shutil.rmtree(db_path)
        >>> os.makedirs(db_path + "test_data/QR100_A/Day1")
        >>> os.makedirs(db_path + "test_data/QR100_A/Day2")
        >>> os.makedirs(db_path + "test_dataset")
        >>> table = ImageTable(db_path + "test_dataset", ["grevy's zebra"])
        >>> add_images(table, [], doctest_mode=True)
        [pipeline] add_images
        [pipeline] len(gpath_list) = 0
        [pipeline] No images to load: exiting...
        []
        >>> prng = RandomState(0)
        >>> for n in range(2):
        ...     a = prng.rand(30, 30, 3) * 255
        ...     img = Image.fromarray(a.astype('uint8')).convert('RGB')
        ...     img.save(db_path + ('test_data/QR100_A/Day1/img%000d.jpg' % n))
        >>> for n in range(2):
        ...     a = prng.rand(30, 30, 3) * 255
        ...     img = Image.fromarray(a.astype('uint8')).convert('RGB')
        ...     img.save(db_path + ('test_data/QR100_A/Day2/img%000d.jpg' % n))
        >>> gpath_list = [db_path + 'test_data/QR100_A/Day1/img0.jpg', db_path + 'test_data/QR100_A/Day1/img1.jpg',
        ...               db_path + 'test_data/QR100_A/Day2/img0.jpg', db_path + 'test_data/QR100_A/Day2/img1.jpg']
        >>> add_images(table, gpath_list, doctest_mode=True)
        [pipeline] add_images
        [pipeline] len(gpath_list) = 4
        [parse_imageinfo] parsing images [1/4]
        [parse_imageinfo] parsing images [2/4]
        [parse_imageinfo] parsing images [3/4]
        [parse_imageinfo] parsing images [4/4]
        Adding 4 image records to DB
            ...added 4 image rows to DB (4 unique)
        Localizing doctest_files/test_data/QR100_A/Day1/img0.jpg -> doctest_files/test_dataset/images/df53d013-889f-e6bf-2636-764a0cd2ce72.jpg
            ...image copied
        Localizing doctest_files/test_data/QR100_A/Day1/img1.jpg -> doctest_files/test_dataset/images/9320f5c0-adf7-2b93-632e-c5537a7ffd15.jpg
            ...image copied
        Localizing doctest_files/test_data/QR100_A/Day2/img0.jpg -> doctest_files/test_dataset/images/56e735a5-53c4-a2a2-428d-8b4fc8933a9d.jpg
            ...image copied
        Localizing doctest_files/test_data/QR100_A/Day2/img1.jpg -> doctest_files/test_dataset/images/633f24d1-fe31-a6fe-4f05-ebb012efa99e.jpg
            ...image copied
        checking image loadable
        [check_image_loadable] validating images [1/4]
        [check_image_loadable] validating images [2/4]
        [check_image_loadable] validating images [3/4]
        [check_image_loadable] validating images [4/4]
            ...validated 4 image rows in DB (4 unique)
        checking image depth
        [check_image_bit_depth] checking image bit depth [1/4]
        [check_image_bit_depth] checking image bit depth [2/4]
        [check_image_bit_depth] checking image bit depth [3/4]
        [check_image_bit_depth] checking image bit depth [4/4]
        [check_image_bit_depth] updated 0 images
        [1, 2, 3, 4]
        >>> print(len([name for name in os.listdir(db_path + "test_dataset/images")
        ...            if os.path.isfile(os.path.join(db_path + "test_dataset/images", name)) and name[-3:] == "jpg"]))
        4
        >>> shutil.rmtree(db_path + "test_data")
        >>> shutil.rmtree(db_path + "test_dataset")
    """

    print(f"[pipeline] add_images")
    print(f"[pipeline] len(gpath_list) = {len(gpath_list)}")
    if len(gpath_list) == 0:
        print(f"[pipeline] No images to load: exiting...")
        return []

    # Create database directory if it doesn't
    Path(imgtable.imgdir).mkdir(parents=True, exist_ok=True)
    Path(imgtable.trashdir).mkdir(parents=True, exist_ok=True)

    compute_params = params_list is None

    if compute_params:
        params_list = _compute_image_params(gpath_list, doctest_mode=doctest_mode)

    colnames = IMAGE_COLNAMES + ("original_path", "location_code")
    params_list = [
        tuple(params) + (gpath, location_for_names) if params is not None else None
        for params, gpath in zip(params_list, gpath_list)
    ]

    print(f"Adding {len(params_list)} image records to DB")
    all_gid_list = imgtable.add_image_data(colnames, params_list)
    print(
        f"\t...added {len(all_gid_list)} image rows to DB ({len(set(all_gid_list))} unique)"
    )

    # Filter for valid images and de-duplicate
    none_set = {None}
    all_gid_set = set(all_gid_list)
    all_valid_gid_set = all_gid_set - none_set
    all_valid_gid_list = list(all_valid_gid_set)

    if auto_localize:
        # Move to wbia database local cache
        localize_images(imgtable, all_valid_gid_list)

    # Check loadable

    ensure_loadable = False  # Temporarily disable loadable check
    if ensure_loadable:
        valid_gpath_list = imgtable.get_image_paths(all_valid_gid_list)
        bad_load_list = check_image_loadable(
            imgtable, all_valid_gid_list, doctest_mode=doctest_mode
        )
        bad_load_set = set(bad_load_list)

        delete_gid_set = set()
        for valid_gid, valid_gpath in zip(all_valid_gid_list, valid_gpath_list):
            if ensure_loadable and valid_gid in bad_load_set:
                print(
                    "Loadable Image Validation: Failed to load {!r}".format(valid_gpath)
                )
                delete_gid_set.add(valid_gid)

        delete_gid_list = list(delete_gid_set)
        if len(delete_gid_list) > 0:
            imgtable.delete_images(delete_gid_list, trash_images=False)

        all_valid_gid_set = all_gid_set - delete_gid_set - none_set
        all_valid_gid_list = list(all_valid_gid_set)

    print(
        f"\t...validated {len(all_valid_gid_list)} image rows in DB ({len(set(all_valid_gid_list))} unique)"
    )

    if not compute_params:
        # We need to double check that the UUIDs are valid, considering we received the UUIDs
        guuid_list = imgtable.get_image_uuids(all_gid_list)
        guuid_list_ = compute_image_uuids(gpath_list)
        assert guuid_list == guuid_list_

    # if as_annots:
    #     # Add succesfull imports as annotations
    #     aid_list = ibs.use_images_as_annotations(all_valid_gid_list)
    #     print(f'[ibs] added {len(aid_list)} annotations')

    # None out any gids that didn't pass the validity check
    assert None not in all_valid_gid_set
    all_gid_list = [aid if aid in all_valid_gid_set else None for aid in all_gid_list]
    assert len(imgtable.get_image_uris_original(get_valid_gids(imgtable))) == len(
        all_gid_list
    )
    if len(all_gid_list) == 0:
        return all_gid_list

    # Ensure all images that were added are 8-bit images
    check_image_bit_depth(
        imgtable, gid_list=list(set(all_gid_list)), doctest_mode=doctest_mode
    )

    return all_gid_list
