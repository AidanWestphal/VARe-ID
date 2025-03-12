"""
developer convenience functions
"""

import os.path

from control.image_funcs import add_images
from db.directory import Directory
from db.table import ImageTable


# Import images recursively from path into database
def import_folder(dir_in, dir_out, file_out, recursive=True, doctest_mode=False):
    """Import images recursively from path into image table.

    Parameters:
        dir_in (str): directory to import from
        dir_out (str): directory to store localized images and output files
        file_out (str): name of output file to save image metadata
        recursive (bool): whether or not to search for images recursively in dir_in
        doctest_mode (bool): if true, replaces carriage returns with newlines

    Returns:
        gid_list (list): list of gids of images that were imported

    Doctest Command:
        python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE control/con_funcs.py

    Example:
        >>> import os
        >>> import numpy
        >>> import shutil
        >>> from numpy.random import RandomState
        >>> from PIL import Image
        >>> db_path = "doctest_files/"
        >>> os.makedirs(db_path + "test_data/QR100_A/Day1")
        >>> os.makedirs(db_path + "test_data/QR100_A/Day2")
        >>> os.makedirs(db_path + "test_dataset")
        >>> import_folder(db_path + "test_data/", db_path + "test_dataset", "image_data.json", doctest_mode=True)
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
        >>> import_folder(db_path + "test_data/", db_path + "test_dataset/", "image_data.json", doctest_mode=True) # doctest: +ELLIPSIS
        [pipeline] add_images
        [pipeline] len(gpath_list) = 4
        [parse_imageinfo] parsing images [1/4]
        [parse_imageinfo] parsing images [2/4]
        [parse_imageinfo] parsing images [3/4]
        [parse_imageinfo] parsing images [4/4]
        Adding 4 image records to DB
            ...added 4 image rows to DB (4 unique)
        Localizing ...doctest_files/test_data/QR100_A/Day1/img0.jpg -> doctest_files/test_dataset/images/df53d013-889f-e6bf-2636-764a0cd2ce72.jpg
            ...image copied
        Localizing ...doctest_files/test_data/QR100_A/Day1/img1.jpg -> doctest_files/test_dataset/images/9320f5c0-adf7-2b93-632e-c5537a7ffd15.jpg
            ...image copied
        Localizing ...doctest_files/test_data/QR100_A/Day2/img0.jpg -> doctest_files/test_dataset/images/56e735a5-53c4-a2a2-428d-8b4fc8933a9d.jpg
            ...image copied
        Localizing ...doctest_files/test_data/QR100_A/Day2/img1.jpg -> doctest_files/test_dataset/images/633f24d1-fe31-a6fe-4f05-ebb012efa99e.jpg
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
        [table.export_to_json] Exporting table to json...
            ...exported 4 image records
        [1, 2, 3, 4]
        >>> print(len([name for name in os.listdir(db_path + "test_dataset/images")
        ...            if os.path.isfile(os.path.join(db_path + "test_dataset/images", name)) and name[-3:] == "jpg"]))
        4
        >>> os.path.exists(db_path + "test_dataset/image_data.json")
        True
        >>> shutil.rmtree(db_path + "test_data")
        >>> shutil.rmtree(db_path + "test_dataset")
    """

    path_out = os.path.join(dir_out, file_out.replace("/", ""))
    direct = Directory(dir_in, recursive=recursive, images=True)
    imgtable = ImageTable(dir_out, ["None"])
    files = direct.files()

    # Load image data from json when applicable
    if os.path.isfile(path_out) and os.path.getsize(path_out) != 0:
        imgtable = ImageTable(dir_out)
        imgtable.import_from_json(path_out)
        gid_list = imgtable.get_all_gids()
        uri_set = set(imgtable.get_image_uris_original(gid_list))
        file_set = set(direct.files())
        files = list(file_set - uri_set)

    # Add images to database
    gid_list = add_images(imgtable, files, doctest_mode=doctest_mode)
    if gid_list:
        imgtable.export_to_json(path_out)
    return gid_list
