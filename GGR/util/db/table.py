import json
from os.path import isabs, join
from PIL import Image
from shapely.geometry import Point, Polygon

import GGR.util.utils as ut
from GGR.util.constants import VALID_COLNAMES
from GGR.util.preproc import get_exif


class ImageTable:
    """
    A class representing a container for image metadata.

    Attributes:
        name (str): The name of the employee.
        table (dict): The primary container for image data
        imgdir (str): The directory images are localized to
        trashdir (str): The directory images are discarded to
        categories (list): List specifying the species contained in images within the table
    """

    def __init__(self, dir, categories=[]):
        """
        Initializes an ImageTable object.

        Parameters:
            dir (str): the database directory
            categories (list): List specifying the species contained in images within the table

        Doctest Command:
            python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE db/table.py

        Example:
            >>> table = ImageTable("test_data/", ["grevy's zebra"])
            >>> print(f"{table.table}")
            {'gid': []}
            >>> print(f"{table.imgdir} {table.trashdir} {table.categories}")
            test_data/images test_data/trash ["grevy's zebra"]
        """

        # 'gid' serves as primary identification for images within database
        self.table = {"gid": []}
        self.imgdir = join(dir, "images")
        self.trashdir = join(dir, "trash")
        self.categories = categories

    def add_image_columns(self, colnames, params_list):
        """
        Adds columns to image table and initializes table with new images if there are none.
        Does not add new images to image table outside of initialization.

        Parameters:
            colnames (list): list of new column names to add
            params_list (list): respective list of image data for each column in colnames

        Returns:
            gid_list (list): list of gids of all images in the table

        Doctest Command:
            python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE db/table.py

        Example:
            >>> table = ImageTable("test_data/", ["grevy's zebra"])
            >>> table.add_image_columns(['uuid', 'uri', 'original_name', 'ext'],
            ... [['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', '/data/GGR2020/QR100_A/Day1/DSCN5265.JPG', 'DSCN5265', '.JPG'],
            ... ['11ca4e9b-1de9-4ee5-8d90-615bf93077c8', '/data/GGR2020/QR100_A/Day1/DSCN5266.JPG', 'DSCN5266', '.JPG'],
            ... ['11ca4e9b-1de9-4ee5-8d90-615bf93077c8', '/data/GGR2020/QR100_A/Day1/DSCN5266.JPG', 'DSCN5266', '.JPG']])
            [table.add_image_columns] Skipping 1 duplicate image(s)...
            [1, 2]
            >>> print(table.table)
            {'gid': [1, 2], 'uuid': ['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', '11ca4e9b-1de9-4ee5-8d90-615bf93077c8'],
             'uri': ['/data/GGR2020/QR100_A/Day1/DSCN5265.JPG', '/data/GGR2020/QR100_A/Day1/DSCN5266.JPG'],
             'original_name': ['DSCN5265', 'DSCN5266'], 'ext': ['.JPG', '.JPG']}
            >>> table.add_image_columns(['fake_column', 'ext', 'width', 'height'],
            ... [['fake', '.JPG', 4608, 3456], ['fake', '.JPG', 4608, 3456]])
            [table.add_image_columns] Invalid column name (fake_column): skipping column...
            [table.add_image_columns] Duplicate column name (ext): skipping column...
            [1, 2]
            >>> print(table.table)
            {'gid': [1, 2], 'uuid': ['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', '11ca4e9b-1de9-4ee5-8d90-615bf93077c8'],
             'uri': ['/data/GGR2020/QR100_A/Day1/DSCN5265.JPG', '/data/GGR2020/QR100_A/Day1/DSCN5266.JPG'],
             'original_name': ['DSCN5265', 'DSCN5266'], 'ext': ['.JPG', '.JPG'], 'width': [4608, 4608], 'height': [3456, 3456]}
        """

        colnames, params_list = list(colnames), list(params_list)
        # Initialize table
        if not self.table["gid"]:
            assert "uuid" in colnames, "UUID must be specified for new images"
            col_idx = colnames.index("uuid")

            dup_ct = 0
            to_remove = []

            # Ensure images added are unique
            for row_idx in range(1, len(params_list)):
                for row_idx2 in range(row_idx):
                    if params_list[row_idx][col_idx] == params_list[row_idx2][col_idx]:
                        dup_ct += 1
                        to_remove.append(row_idx)
                        break

            for row_idx in reversed(to_remove):
                params_list.pop(row_idx)

            if dup_ct:
                print(
                    f"[table.add_image_columns] Skipping {dup_ct} duplicate image(s)..."
                )

            # Automatically assign gids to new images
            self.table["gid"] = list(range(1, len(params_list) + 1))

        # Add data to table for each column provided
        for col_idx in range(len(colnames)):
            colname = colnames[col_idx]
            # Ensure current column is valid, unique, and equivalent in size to other columns
            if colname not in VALID_COLNAMES:
                print(
                    f"[table.add_image_columns] Invalid column name ({colname}): skipping column..."
                )
                continue
            elif colname in self.table.keys():
                print(
                    f"[table.add_image_columns] Duplicate column name ({colname}): skipping column..."
                )
                continue

            # Update table
            self.table[colname] = []
            for row_idx in range(len(params_list)):
                self.table[colname].append(params_list[row_idx][col_idx])

        return self.table["gid"]

    def add_image_rows(self, colnames, params_list):
        """
        Adds rows (images) to image table.
        Does not add new columns to image table.
        Does not initialize table when there are no images.

        Parameters:
            colnames (list): list of existing column names (must match table keys)
            params_list (list): respective list of image data for each column in colnames

        Returns:
            gid_list (list): list of gids of all images in the table

        Doctest Command:
            python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE db/table.py

        Example:
            >>> table = ImageTable("test_data/", ["grevy's zebra"])
            >>> table.add_image_columns(['uuid', 'uri', 'original_name', 'ext'],
            ... [['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', '/data/GGR2020/QR100_A/Day1/DSCN5265.JPG', 'DSCN5265', '.JPG'],
            ... ['11ca4e9b-1de9-4ee5-8d90-615bf93077c8', '/data/GGR2020/QR100_A/Day1/DSCN5266.JPG', 'DSCN5266', '.JPG']])
            [1, 2]
            >>> table.add_image_rows(['uuid', 'uri', 'original_name', 'ext'],
            ... [['11ca4e9b-1de9-4ee5-8d90-615bf93077c8', '/data/GGR2020/QR100_A/Day1/DSCN5266.JPG', 'DSCN5266', '.JPG'],
            ... ['6830cd43-ca00-28ee-8843-fe64d789d7f7', '/data/GGR2020/QR100_A/Day1/DSCN5267.JPG', 'DSCN5267', '.JPG'],
            ... ['6830cd43-ca00-28ee-8843-fe64d789d7f7', '/data/GGR2020/QR100_A/Day1/DSCN5267.JPG', 'DSCN5267', '.JPG'],
            ... ['b54193f2-8507-8441-5bf3-cd9f0ed8a883', '/data/GGR2020/QR100_A/Day1/DSCN5268.JPG', 'DSCN5268', '.JPG']])
            [add_image_rows] Failed to add 1 existing images to image table
            [add_image_rows] Failed to add 1 duplicate images to image table
            [1, 2, 3, 4]
            >>> print(table.table)
            {'gid': [1, 2, 3, 4], 'uuid': ['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', '11ca4e9b-1de9-4ee5-8d90-615bf93077c8',
             '6830cd43-ca00-28ee-8843-fe64d789d7f7', 'b54193f2-8507-8441-5bf3-cd9f0ed8a883'],
             'uri': ['/data/GGR2020/QR100_A/Day1/DSCN5265.JPG', '/data/GGR2020/QR100_A/Day1/DSCN5266.JPG',
             '/data/GGR2020/QR100_A/Day1/DSCN5267.JPG', '/data/GGR2020/QR100_A/Day1/DSCN5268.JPG'],
             'original_name': ['DSCN5265', 'DSCN5266', 'DSCN5267', 'DSCN5268'], 'ext': ['.JPG', '.JPG', '.JPG', '.JPG']}
            >>> table.add_image_rows(['uuid', 'uri', 'original_name', 'ext'], [])
            [1, 2, 3, 4]
        """

        if not colnames or not params_list:
            return self.table["gid"]

        colnames, params_list = list(colnames), list(params_list)
        colnames_test = list(colnames)
        colnames_test.append("gid")

        # Ensure colnames are identical to columns in table
        if sorted(colnames_test) != sorted(self.table.keys()):
            print(
                "[add_image_rows] Provided columns are not identical to table columns: aborting operation..."
            )
            return self.table["gid"]

        # Ensure images added are new & unique
        for col_idx in range(len(colnames)):
            if colnames[col_idx] == "uuid":
                old_ct = 0
                dup_ct = 0
                to_remove = []

                if params_list[0][col_idx] in self.table["uuid"]:
                    old_ct += 1
                    to_remove.append(0)

                for row_idx in range(1, len(params_list)):
                    if params_list[row_idx][col_idx] in self.table["uuid"]:
                        old_ct += 1
                        to_remove.append(row_idx)
                        continue

                    for row_idx2 in range(row_idx):
                        if (
                            params_list[row_idx][col_idx]
                            == params_list[row_idx2][col_idx]
                        ):
                            dup_ct += 1
                            to_remove.append(row_idx)
                            break

                for row_idx in reversed(to_remove):
                    params_list.pop(row_idx)

                if old_ct:
                    print(
                        f"[add_image_rows] Failed to add {old_ct} existing images to image table"
                    )
                if dup_ct:
                    print(
                        f"[add_image_rows] Failed to add {dup_ct} duplicate images to image table"
                    )

        # Automatically assign gids to new images
        num_images = len(self.table["gid"])
        self.table["gid"].extend(
            list(range(num_images + 1, num_images + len(params_list) + 1))
        )
        # Add data to table for each column provided
        for col_idx in range(len(colnames)):
            for row_idx in range(len(params_list)):
                self.table[colnames[col_idx]].append(params_list[row_idx][col_idx])

        return self.table["gid"]

    def add_image_data(self, colnames, params_list):
        """
        Adds columns or rows (images) to image table.
        Uses add_image_columns or add_image_rows.

        Parameters:
            colnames (list): list of new or existing column names
            params_list (list): respective list of image data for each column in colnames

        Returns:
            gid_list (list): list of gids of all images in the table

        Doctest Command:
            python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE db/table.py

        Example:
            >>> table = ImageTable("test_data/", ["grevy's zebra"])
            >>> table.add_image_data(['uuid', 'uri', 'original_name', 'ext'],
            ... [['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', '/data/GGR2020/QR100_A/Day1/DSCN5265.JPG', 'DSCN5265', '.JPG'],
            ... ['11ca4e9b-1de9-4ee5-8d90-615bf93077c8', '/data/GGR2020/QR100_A/Day1/DSCN5266.JPG', 'DSCN5266', '.JPG']])
            [1, 2]
            >>> table.add_image_data(['uuid', 'uri', 'original_name', 'ext'],
            ... [['11ca4e9b-1de9-4ee5-8d90-615bf93077c8', '/data/GGR2020/QR100_A/Day1/DSCN5266.JPG', 'DSCN5266', '.JPG'],
            ... ['6830cd43-ca00-28ee-8843-fe64d789d7f7', '/data/GGR2020/QR100_A/Day1/DSCN5267.JPG', 'DSCN5267', '.JPG'],
            ... ['6830cd43-ca00-28ee-8843-fe64d789d7f7', '/data/GGR2020/QR100_A/Day1/DSCN5267.JPG', 'DSCN5267', '.JPG'],
            ... ['b54193f2-8507-8441-5bf3-cd9f0ed8a883', '/data/GGR2020/QR100_A/Day1/DSCN5268.JPG', 'DSCN5268', '.JPG']])
            [add_image_rows] Failed to add 1 existing images to image table
            [add_image_rows] Failed to add 1 duplicate images to image table
            [1, 2, 3, 4]
            >>> print(table.table)
            {'gid': [1, 2, 3, 4], 'uuid': ['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', '11ca4e9b-1de9-4ee5-8d90-615bf93077c8',
             '6830cd43-ca00-28ee-8843-fe64d789d7f7', 'b54193f2-8507-8441-5bf3-cd9f0ed8a883'],
             'uri': ['/data/GGR2020/QR100_A/Day1/DSCN5265.JPG', '/data/GGR2020/QR100_A/Day1/DSCN5266.JPG',
             '/data/GGR2020/QR100_A/Day1/DSCN5267.JPG', '/data/GGR2020/QR100_A/Day1/DSCN5268.JPG'],
             'original_name': ['DSCN5265', 'DSCN5266', 'DSCN5267', 'DSCN5268'], 'ext': ['.JPG', '.JPG', '.JPG', '.JPG']}
            >>> table.add_image_data(['uuid', 'uri', 'original_name', 'ext'], [])
            [1, 2, 3, 4]
        """

        # Add rows if table is initialized and no new columns are added
        # Add columns if no provided columns exist in the table
        colnames_test = list(colnames)
        colnames_test.append("gid")

        if self.table["gid"] and sorted(colnames_test) == sorted(self.table.keys()):
            return self.add_image_rows(colnames, params_list)
        elif all([colname not in self.table.keys() for colname in colnames]):
            return self.add_image_columns(colnames, params_list)
        else:
            print("[add_image_data] Invalid columns input: aborting operation...")
            return self.table["gid"]

    def delete_rowids(self, gid_list):
        """
        Removes rows from image table specified by gids.

        Parameters:
            gid_list (list): list of gids

        Returns:
            gid_list (list): list of gids of all images in the table

        Doctest Command:
            python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE db/table.py

        Example:
            >>> table = ImageTable("test_data/", ["grevy's zebra"])
            >>> table.add_image_data(['uuid', 'uri', 'original_name', 'ext'],
            ... [['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', '/data/GGR2020/QR100_A/Day1/DSCN5265.JPG', 'DSCN5265', '.JPG'],
            ... ['11ca4e9b-1de9-4ee5-8d90-615bf93077c8', '/data/GGR2020/QR100_A/Day1/DSCN5266.JPG', 'DSCN5266', '.JPG'],
            ... ['6830cd43-ca00-28ee-8843-fe64d789d7f7', '/data/GGR2020/QR100_A/Day1/DSCN5267.JPG', 'DSCN5267', '.JPG'],
            ... ['b54193f2-8507-8441-5bf3-cd9f0ed8a883', '/data/GGR2020/QR100_A/Day1/DSCN5268.JPG', 'DSCN5268', '.JPG']])
            [1, 2, 3, 4]
            >>> table.delete_rowids([1, 3])
            >>> print(table.table['gid'])
            [2, 4]
        """

        for gid in reversed(sorted(gid_list)):
            for colname in self.table.keys():
                self.table[colname].pop(gid - 1)

    def delete_images(self, gid_list, trash_images=True):
        """
        Removes rows from image table specified by gids and their respective images from imgdir.
        Moves specified images to trashdir instead if trash_images is true.
        Realigns gids for remaining image rows.

        Parameters:
            gid_list (list): list of gids

        Returns:
            gid_list (list): list of gids of all images in the table

        Doctest Command:
            python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE db/table.py

        Example:
            >>> from PIL import Image
            >>> import os
            >>> import shutil
            >>> db_path = 'doctest_data/'
            >>> if os.path.exists(db_path) and os.path.isdir(db_path):
            ...     shutil.rmtree(db_path)
            >>> os.makedirs(db_path + 'images')
            >>> img = Image.new('RGB',(480,640),'rgb(255,255,255)')
            >>> img.save(db_path + 'images/0fdef8e8-cec0-b460-bac2-6ee3e39f0798.JPG')
            >>> img = Image.new('RGB',(480,640),'rgb(0,0,0)')
            >>> img.save(db_path + 'images/6830cd43-ca00-28ee-8843-fe64d789d7f7.JPG')
            >>> table = ImageTable('doctest_data', ["grevy's zebra"])
            >>> table.add_image_data(['uuid', 'uri', 'original_name', 'ext'],
            ... [['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', '0fdef8e8-cec0-b460-bac2-6ee3e39f0798.JPG', 'DSCN5265', '.JPG'],
            ... ['11ca4e9b-1de9-4ee5-8d90-615bf93077c8', '11ca4e9b-1de9-4ee5-8d90-615bf93077c8.JPG', 'DSCN5266', '.JPG'],
            ... ['6830cd43-ca00-28ee-8843-fe64d789d7f7', '6830cd43-ca00-28ee-8843-fe64d789d7f7.JPG', 'DSCN5267', '.JPG'],
            ... ['b54193f2-8507-8441-5bf3-cd9f0ed8a883', 'b54193f2-8507-8441-5bf3-cd9f0ed8a883.JPG', 'DSCN5268', '.JPG']])
            [1, 2, 3, 4]
            >>> table.delete_images([1, 3])
            [table.delete_images] deleting 2 images
            [ensuredir] mkdir('doctest_data/trash')
            [utils.remove_file] Finished deleting path='doctest_data/images/0fdef8e8-cec0-b460-bac2-6ee3e39f0798.JPG'
            [utils.remove_file] Finished deleting path='doctest_data/images/6830cd43-ca00-28ee-8843-fe64d789d7f7.JPG'
            [1, 2]
            >>> os.path.exists(db_path + 'images/0fdef8e8-cec0-b460-bac2-6ee3e39f0798.JPG')
            False
            >>> os.path.exists(db_path + 'images/6830cd43-ca00-28ee-8843-fe64d789d7f7.JPG')
            False
            >>> os.path.exists(db_path + 'trash/DSCN5265')
            True
            >>> os.path.exists(db_path + 'trash/DSCN5267')
            True
            >>> ut.remove_file(db_path + 'trash/DSCN5265')
            [utils.remove_file] Finished deleting path='doctest_data/trash/DSCN5265'
            True
            >>> ut.remove_file(db_path + 'trash/DSCN5267')
            [utils.remove_file] Finished deleting path='doctest_data/trash/DSCN5267'
            True
        """

        print(f"[table.delete_images] deleting {len(gid_list)} images")
        # Move images to trash before deleting them.
        gpath_list = self.get_image_paths(gid_list)
        gname_list = self.get_image_gnames(gid_list)
        ext_list = self.get_image_exts(gid_list)
        if trash_images:
            ut.ensuredir(self.trashdir)
            gpath_list2 = [join(self.trashdir, gname) for gname in gname_list]
            ut.copy_file_list(gpath_list, gpath_list2, err_ok=True)
        for gpath in gpath_list:
            ut.remove_file(gpath)

        # # Delete tiles first, find any tiles that depend on these images as an ancestor
        # descendants_gids_list = ibs.get_tile_descendants_gids(gid_list)
        # descendants_gid_list = list(set(ut.flatten(descendants_gids_list)))
        # if len(descendants_gid_list) > 0:
        #     ibs.delete_images(descendants_gid_list)

        # # Delete annotations second (only for images not tiles)
        # tile_flag_list = ibs.get_tile_flags(gid_list)
        # image_gid_list = ut.filterfalse_items(gid_list, tile_flag_list)
        # aid_list = ut.flatten(ibs.get_image_aids(image_gid_list))
        # if len(aid_list) > 0:
        #     ibs.delete_annots(aid_list)

        # delete thumbs in case an annot doesnt delete them
        gid_list = list(set(gid_list))
        # ibs.delete_image_thumbs(gid_list)
        # ibs.depc_image.delete_root(gid_list)
        self.delete_rowids(gid_list)
        # ibs.db.delete(const.GSG_RELATION_TABLE, gid_list, id_colname='image_rowid')
        self.table["gid"] = list(range(1, len(self.table["gid"]) + 1))

        return self.table["gid"]

    def import_from_json(self, filepath):
        """
        Imports image table from json.

        Parameters:
            filepath (str): path to json file

        Doctest Command:
            python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE db/table.py

        Example:
            >>> db_path = 'doctest_files/'
            >>> json_path = db_path + 'test.json'
            >>> json_dict = {"categories": [{"id": 0, "species": "grevy's zebra"}],
            ... "images": [{"gid": 1, "uuid": "0fdef8e8-cec0-b460-bac2-6ee3e39f0798",
            ... "uri": "/data/GGR2020/QR100_A/Day1/DSCN5265.JPG", "original_name": "DSCN5265", "ext": ".JPG"},
            ... {"gid": 2, "uuid": "11ca4e9b-1de9-4ee5-8d90-615bf93077c8",
            ... "uri": "/data/GGR2020/QR100_A/Day1/DSCN5266.JPG", "original_name": "DSCN5266", "ext": ".JPG"}]}
            >>> with open(json_path, 'w') as outfile:
            ...     json.dump(json_dict, outfile, indent=4)
            >>> table = ImageTable(db_path)
            >>> table.import_from_json(json_path)
            [table] Importing table from json...
                ...found 2 image records
            >>> print(table.table)
            {'gid': [1, 2], 'uuid': ['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', '11ca4e9b-1de9-4ee5-8d90-615bf93077c8'],
             'uri': ['/data/GGR2020/QR100_A/Day1/DSCN5265.JPG', '/data/GGR2020/QR100_A/Day1/DSCN5266.JPG'],
             'original_name': ['DSCN5265', 'DSCN5266'], 'ext': ['.JPG', '.JPG']}
            >>> ut.remove_file(json_path)
            [utils.remove_file] Finished deleting path='doctest_files/test.json'
            True
        """

        print("[table] Importing table from json...")
        with open(filepath, "r") as infile:
            in_dict = json.load(infile)

        for cat in in_dict["categories"]:
            self.categories.append(cat["species"])

        for img_data in in_dict["images"]:
            for colname in img_data.keys():
                if colname not in self.table.keys():
                    self.table[colname] = []

                self.table[colname].append(img_data[colname])

        print(f"\t...found {len(self.table['gid'])} image records")

    def export_to_json(self, filepath):
        """
        Exports image table to json.

        Parameters:
            filepath (str): path to json file

        Doctest Command:
            python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE db/table.py

        Example:
            >>> db_path = 'doctest_files/'
            >>> json_path = db_path + 'test.json'
            >>> table = ImageTable(db_path, ["grevy's zebra"])
            >>> table.add_image_data(['uuid', 'uri', 'original_name', 'ext'],
            ... [['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', '/data/GGR2020/QR100_A/Day1/DSCN5265.JPG', 'DSCN5265', '.JPG'],
            ... ['11ca4e9b-1de9-4ee5-8d90-615bf93077c8', '/data/GGR2020/QR100_A/Day1/DSCN5266.JPG', 'DSCN5266', '.JPG']])
            [1, 2]
            >>> table.export_to_json(json_path)
            [table.export_to_json] Exporting table to json...
                ...exported 2 image records
            >>> with open(json_path, 'r') as infile:
            ...     in_dict = json.load(infile)
            >>> json_dict = {"categories": [{"id": 0, "species": "grevy's zebra"}],
            ... "images": [{"gid": 1, "uuid": "0fdef8e8-cec0-b460-bac2-6ee3e39f0798",
            ... "uri": "/data/GGR2020/QR100_A/Day1/DSCN5265.JPG", "original_name": "DSCN5265", "ext": ".JPG"},
            ... {"gid": 2, "uuid": "11ca4e9b-1de9-4ee5-8d90-615bf93077c8",
            ... "uri": "/data/GGR2020/QR100_A/Day1/DSCN5266.JPG", "original_name": "DSCN5266", "ext": ".JPG"}]}
            >>> print(in_dict == json_dict)
            True
        """

        print("[table.export_to_json] Exporting table to json...")
        out_dict = {"categories": [], "images": []}
        for cat_idx in range(len(self.categories)):
            out_dict["categories"].append(
                {"id": cat_idx, "species": self.categories[cat_idx]}
            )

        count = 0
        for img_idx in range(len(self.table["gid"])):
            count += 1
            img_dict = dict()
            for colname in self.table.keys():
                img_dict[colname] = self.table[colname][img_idx]

            out_dict["images"].append(img_dict)

        with open(filepath, "w") as outfile:
            json.dump(out_dict, outfile, indent=4)

        print(f"\t...exported {count} image records")

    def get_categories(self):
        """Gets full species list"""
        return self.categories

    def get_all_gids(self):
        """Gets full gid list"""
        return self.table["gid"]

    def get_imgset_gids(self, imgset_id_list):
        """
        Gets gids of images within a set of image sets

        Parameters:
            imgset_id_list (list): list of image set ids

        Returns:
            imgset_gids_list (list): respective list of gid lists for each image set
        """

        if type(imgset_id_list) == int:
            return self.imgset_list[imgset_id_list - 1]
        else:
            return [
                self.imgset_list[imgset_id - 1] for imgset_id in list(imgset_id_list)
            ]

    def get(self, colnames, gid_list):
        """
        Gets tuples of values from a given tuple of columns in the table corresponding to a list of gids.

        Parameters:
            colnames (list): list of column names
            gid_list (list): list of gids

        Returns:
            data (list): list of corresponding values in the table

        Doctest Command:
            python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE db/table.py

        Example:
            >>> table = ImageTable("test_data/", ["grevy's zebra"])
            >>> gid_list = table.add_image_data(['uuid', 'uri', 'original_name', 'ext'],
            ... [['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', '/data/GGR2020/QR100_A/Day1/DSCN5265.JPG', 'DSCN5265', '.JPG'],
            ... ['11ca4e9b-1de9-4ee5-8d90-615bf93077c8', '/data/GGR2020/QR100_A/Day1/DSCN5266.JPG', 'DSCN5266', '.JPG'],
            ... ['6830cd43-ca00-28ee-8843-fe64d789d7f7', '/data/GGR2020/QR100_A/Day1/DSCN5267.JPG', 'DSCN5267', '.JPG'],
            ... ['b54193f2-8507-8441-5bf3-cd9f0ed8a883', '/data/GGR2020/QR100_A/Day1/DSCN5268.JPG', 'DSCN5268', '.JPG']])
            >>> table.get(('uuid', 'uri', 'original_name', 'ext'), gid_list)
            [('0fdef8e8-cec0-b460-bac2-6ee3e39f0798', '/data/GGR2020/QR100_A/Day1/DSCN5265.JPG', 'DSCN5265', '.JPG'),
             ('11ca4e9b-1de9-4ee5-8d90-615bf93077c8', '/data/GGR2020/QR100_A/Day1/DSCN5266.JPG', 'DSCN5266', '.JPG'),
             ('6830cd43-ca00-28ee-8843-fe64d789d7f7', '/data/GGR2020/QR100_A/Day1/DSCN5267.JPG', 'DSCN5267', '.JPG'),
             ('b54193f2-8507-8441-5bf3-cd9f0ed8a883', '/data/GGR2020/QR100_A/Day1/DSCN5268.JPG', 'DSCN5268', '.JPG')]
            >>> table.get(('original_name', 'ext'), gid_list)
            [('DSCN5265', '.JPG'), ('DSCN5266', '.JPG'), ('DSCN5267', '.JPG'), ('DSCN5268', '.JPG')]
            >>> table.get(('original_name',), gid_list)
            ['DSCN5265', 'DSCN5266', 'DSCN5267', 'DSCN5268']
            >>> table.get(('original_name', 'ext'), gid_list[:2])
            [('DSCN5265', '.JPG'), ('DSCN5266', '.JPG')]
            >>> table.get(('original_name', 'ext'), [gid_list[0]])
            [('DSCN5265', '.JPG')]
            >>> table.get(('original_name', 'ext'), gid_list[0])
            ('DSCN5265', '.JPG')
            >>> table.get(('original_name',), gid_list[:2])
            ['DSCN5265', 'DSCN5266']
            >>> table.get(('original_name',), [gid_list[0]])
            ['DSCN5265']
            >>> table.get(('original_name',), gid_list[0])
            'DSCN5265'
            >>> table.get(tuple(), [])
            []
        """

        if len(colnames) == 0 or (ut.isiterable(gid_list) and len(gid_list) == 0):
            return []

        cols = [self.table[colname] for colname in colnames]
        if len(cols) == 1 and not ut.isiterable(gid_list):
            return cols[0][gid_list - 1]
        elif len(cols) == 1:
            return [cols[0][gid - 1] for gid in gid_list]
        elif not ut.isiterable(gid_list):
            return tuple(cols[col_idx][gid_list - 1] for col_idx in range(len(cols)))
        else:
            return [
                tuple(cols[col_idx][gid - 1] for col_idx in range(len(cols)))
                for gid in gid_list
            ]

    def get_image_uuids(self, gid_list):
        """
        Gets list of uuids corresponding to given gids.
        Gets single uuid if provided with one gid.

        Parameters:
            gid_list (list): list of gids

        Returns:
            uuid_list (list): list of corresponding uuids

        Doctest Command:
            python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE db/table.py

        Example:
            >>> table = ImageTable("test_table/", ["grevy's zebra"])
            >>> gid_list = table.add_image_data(['uuid'],
            ... [['0fdef8e8-cec0-b460-bac2-6ee3e39f0798']])
            >>> table.get_image_uuids(gid_list[0])
            '0fdef8e8-cec0-b460-bac2-6ee3e39f0798'
            >>> gid_list = table.add_image_data(['uuid'],
            ... [['11ca4e9b-1de9-4ee5-8d90-615bf93077c8'],
            ... ['6830cd43-ca00-28ee-8843-fe64d789d7f7'],
            ... ['b54193f2-8507-8441-5bf3-cd9f0ed8a883']])
            >>> table.get_image_uuids(gid_list)
            ['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', '11ca4e9b-1de9-4ee5-8d90-615bf93077c8',
            '6830cd43-ca00-28ee-8843-fe64d789d7f7', 'b54193f2-8507-8441-5bf3-cd9f0ed8a883']
        """

        uuid_list = self.get(("uuid",), gid_list)
        return uuid_list

    def get_image_uris(self, gid_list):
        """
        Gets list of uris corresponding to given gids.
        Gets single uri if provided with one gid.

        Parameters:
            gid_list (list): list of gids

        Returns:
            uri_list (list): list of corresponding uris

        Doctest Command:
            python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE db/table.py

        Example:
            >>> table = ImageTable("test_data/", ["grevy's zebra"])
            >>> gid_list = table.add_image_data(['uuid', 'uri'],
            ... [['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', '/data/GGR2020/QR100_A/Day1/DSCN5265.JPG']])
            >>> table.get_image_uris(gid_list[0])
            '/data/GGR2020/QR100_A/Day1/DSCN5265.JPG'
            >>> gid_list = table.add_image_data(['uuid', 'uri'],
            ... [['11ca4e9b-1de9-4ee5-8d90-615bf93077c8', '/data/GGR2020/QR100_A/Day1/DSCN5266.JPG'],
            ... ['6830cd43-ca00-28ee-8843-fe64d789d7f7', '/data/GGR2020/QR100_A/Day1/DSCN5267.JPG'],
            ... ['b54193f2-8507-8441-5bf3-cd9f0ed8a883', '/data/GGR2020/QR100_A/Day1/DSCN5268.JPG']])
            >>> table.get_image_uris(gid_list)
            ['/data/GGR2020/QR100_A/Day1/DSCN5265.JPG', '/data/GGR2020/QR100_A/Day1/DSCN5266.JPG',
             '/data/GGR2020/QR100_A/Day1/DSCN5267.JPG', '/data/GGR2020/QR100_A/Day1/DSCN5268.JPG']
        """

        uri_list = self.get(("uri",), gid_list)
        return uri_list

    def get_image_uris_original(self, gid_list):
        """
        Gets list of original uris corresponding to given gids.
        Gets single original uri if provided with one gid.

        Parameters:
            gid_list (list): list of gids

        Returns:
            uri_list (list): list of corresponding original uris

        Doctest Command:
            python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE db/table.py

        Example:
            >>> table = ImageTable("test_data/", ["grevy's zebra"])
            >>> gid_list = table.add_image_data(['uuid', 'uri_original'],
            ... [['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', '/data/GGR2020/QR100_A/Day1/DSCN5265.JPG']])
            >>> table.get_image_uris_original(gid_list[0])
            '/data/GGR2020/QR100_A/Day1/DSCN5265.JPG'
            >>> gid_list = table.add_image_data(['uuid', 'uri_original'],
            ... [['11ca4e9b-1de9-4ee5-8d90-615bf93077c8', '/data/GGR2020/QR100_A/Day1/DSCN5266.JPG'],
            ... ['6830cd43-ca00-28ee-8843-fe64d789d7f7', '/data/GGR2020/QR100_A/Day1/DSCN5267.JPG'],
            ... ['b54193f2-8507-8441-5bf3-cd9f0ed8a883', '/data/GGR2020/QR100_A/Day1/DSCN5268.JPG']])
            >>> table.get_image_uris_original(gid_list)
            ['/data/GGR2020/QR100_A/Day1/DSCN5265.JPG', '/data/GGR2020/QR100_A/Day1/DSCN5266.JPG',
             '/data/GGR2020/QR100_A/Day1/DSCN5267.JPG', '/data/GGR2020/QR100_A/Day1/DSCN5268.JPG']
        """

        uri_list = self.get(("uri_original",), gid_list)
        return uri_list

    def get_image_gnames(self, gid_list):
        """
        Gets list of original image names corresponding to given gids.
        Gets single gname if provided with one gid.

        Parameters:
            gid_list (list): list of gids

        Returns:
            gname_list (list): list of corresponding original image names

        Doctest Command:
            python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE db/table.py

        Example:
            >>> table = ImageTable("test_data/", ["grevy's zebra"])
            >>> gid_list = table.add_image_data(['uuid', 'original_name'],
            ... [['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', 'DSCN5265']])
            >>> table.get_image_gnames(gid_list[0])
            'DSCN5265'
            >>> gid_list = table.add_image_data(['uuid', 'original_name'],
            ... [['11ca4e9b-1de9-4ee5-8d90-615bf93077c8', 'DSCN5266'],
            ... ['6830cd43-ca00-28ee-8843-fe64d789d7f7', 'DSCN5267'],
            ... ['b54193f2-8507-8441-5bf3-cd9f0ed8a883', 'DSCN5268']])
            >>> table.get_image_gnames(gid_list)
            ['DSCN5265', 'DSCN5266', 'DSCN5267', 'DSCN5268']
        """

        gname_list = self.get(("original_name",), gid_list)
        return gname_list

    def get_image_exts(self, gid_list):
        """
        Gets list of file extensions corresponding to given gids.
        Gets single ext if provided with one gid.

        Parameters:
            gid_list (list): list of gids

        Returns:
            ext_list (list): list of corresponding file extensions

        Doctest Command:
            python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE db/table.py

        Example:
            >>> table = ImageTable("test_data/", ["grevy's zebra"])
            >>> gid_list = table.add_image_data(['uuid', 'ext'],
            ... [['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', '.jpg']])
            >>> table.get_image_exts(gid_list[0])
            '.jpg'
            >>> gid_list = table.add_image_data(['uuid', 'ext'],
            ... [['11ca4e9b-1de9-4ee5-8d90-615bf93077c8', '.JPG'],
            ... ['6830cd43-ca00-28ee-8843-fe64d789d7f7', '.jpg'],
            ... ['b54193f2-8507-8441-5bf3-cd9f0ed8a883', '.JPG']])
            >>> table.get_image_exts(gid_list)
            ['.jpg', '.JPG', '.jpg', '.JPG']
        """

        ext_list = self.get(("ext",), gid_list)
        return ext_list

    def get_image_dims(self, gid_list):
        """
        Gets list of image dimension data corresponding to given gids.
        Gets single dimension tuple if provided with one gid.

        Parameters:
            gid_list (list): list of gids

        Returns:
            dim_list (list): list of corresponding (width, height) tuples

        Doctest Command:
            python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE db/table.py

        Example:
            >>> table = ImageTable("test_data/", ["grevy's zebra"])
            >>> gid_list = table.add_image_data(['uuid', 'width', 'height'],
            ... [['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', 30, 40]])
            >>> table.get_image_dims(gid_list[0])
            (30, 40)
            >>> gid_list = table.add_image_data(['uuid', 'width', 'height'],
            ... [['11ca4e9b-1de9-4ee5-8d90-615bf93077c8', 40, 50],
            ... ['6830cd43-ca00-28ee-8843-fe64d789d7f7', 50, 60],
            ... ['b54193f2-8507-8441-5bf3-cd9f0ed8a883', 60, 70]])
            >>> table.get_image_dims(gid_list)
            [(30, 40), (40, 50), (50, 60), (60, 70)]
        """

        dim_list = self.get(("width", "height"), gid_list)
        return dim_list

    def get_image_unixtimes(self, gid_list, timedelta_correction=True):
        """
        Gets list of unixtimes corresponding to given gids.
        Gets single time datum if provided with one gid.

        Parameters:
            gid_list (list): list of gids

        Returns:
            unixtime_list (list): list of corresponding times that images were taken
            (-1 if no time data exists for a given gid)

        Doctest Command:
            python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE db/table.py

        Example:
            >>> table = ImageTable("test_data/", ["grevy's zebra"])
            >>> gid_list = table.add_image_data(['uuid', 'time_posix'],
            ... [['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', 1719611896]])
            >>> table.get_image_unixtimes(gid_list[0])
            1719611896
            >>> gid_list = table.add_image_data(['uuid', 'time_posix'],
            ... [['11ca4e9b-1de9-4ee5-8d90-615bf93077c8', 1719611897],
            ... ['6830cd43-ca00-28ee-8843-fe64d789d7f7', 1719611898],
            ... ['b54193f2-8507-8441-5bf3-cd9f0ed8a883', 1719611899]])
            >>> table.get_image_unixtimes(gid_list)
            [1719611896, 1719611897, 1719611898, 1719611899]
        """

        unixtime_list = self.get(("time_posix",), gid_list)
        if ut.isiterable(unixtime_list):
            unixtime_list = [
                -1 if unixtime is None else unixtime for unixtime in unixtime_list
            ]
        else:
            unixtime_list = -1 if unixtime_list is None else unixtime_list

        # if timedelta_correction:
        #     timedelta_list = ibs.get_image_timedelta_posix(gid_list)
        #     timedelta_list = [
        #         0 if timedelta is None else timedelta for timedelta in timedelta_list
        #     ]
        #     unixtime_list = [
        #         unixtime + timedelta
        #         for unixtime, timedelta in zip(unixtime_list, timedelta_list)
        #     ]
        return unixtime_list

    def get_image_gps(self, gid_list):
        """
        Gets list of gps data corresponding to given gids.
        Gets gps datum if provided with one gid.

        Parameters:
            gid_list (list): list of gids

        Returns:
            gps_list (list): list of corresponding gps tuples
            ((-1, -1) if no gps data exists for a given gid)

        Doctest Command:
            python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE db/table.py

        Example:
            >>> table = ImageTable("test_data/", ["grevy's zebra"])
            >>> gid_list = table.add_image_data(['uuid', 'gps_lat', 'gps_lon'],
            ... [['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', 0.291885, 36.89818783]])
            >>> table.get_image_gps(gid_list[0])
            (0.291885, 36.89818783)
            >>> gid_list = table.add_image_data(['uuid', 'gps_lat', 'gps_lon'],
            ... [['11ca4e9b-1de9-4ee5-8d90-615bf93077c8', 1.291885, 37.89818783],
            ... ['6830cd43-ca00-28ee-8843-fe64d789d7f7', 2.291885, 38.89818783],
            ... ['b54193f2-8507-8441-5bf3-cd9f0ed8a883', 3.291885, 39.89818783]])
            >>> table.get_image_gps(gid_list)
            [(0.291885, 36.89818783), (1.291885, 37.89818783), (2.291885, 38.89818783), (3.291885, 39.89818783)]
        """

        gps_list = self.get(("gps_lat", "gps_lon"), gid_list)
        return gps_list

    def get_image_orientations(self, gid_list):
        """
        Gets list of orientations corresponding to given gids.
        Gets single orientation if provided with one gid.

        Parameters:
            gid_list (list): list of gids

        Returns:
            orient_list (list): list of corresponding orientations

        Doctest Command:
            python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE db/table.py

        Example:
            >>> table = ImageTable("test_data/", ["grevy's zebra"])
            >>> gid_list = table.add_image_data(['uuid', 'orientation'],
            ... [['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', 0]])
            >>> table.get_image_orientations(gid_list[0])
            0
            >>> gid_list = table.add_image_data(['uuid', 'orientation'],
            ... [['11ca4e9b-1de9-4ee5-8d90-615bf93077c8', 1],
            ... ['6830cd43-ca00-28ee-8843-fe64d789d7f7', 2],
            ... ['b54193f2-8507-8441-5bf3-cd9f0ed8a883', 3]])
            >>> table.get_image_orientations(gid_list)
            [0, 1, 2, 3]
        """

        orient_list = self.get(("orientation",), gid_list)
        return orient_list

    def get_image_notes(self, gid_list):
        """
        Gets list of notes corresponding to given gids.

        Parameters:
            gid_list (list): list of gids

        Returns:
            note_list (list): list of corresponding notes

        Doctest Command:
            python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE db/table.py

        Example:
            >>> table = ImageTable("test_data/", ["grevy's zebra"])
            >>> gid_list = table.add_image_data(['uuid', 'note'],
            ... [['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', 'note1']])
            >>> table.get_image_notes(gid_list[0])
            'note1'
            >>> gid_list = table.add_image_data(['uuid', 'note'],
            ... [['11ca4e9b-1de9-4ee5-8d90-615bf93077c8', 'note2'],
            ... ['6830cd43-ca00-28ee-8843-fe64d789d7f7', 'note3'],
            ... ['b54193f2-8507-8441-5bf3-cd9f0ed8a883', 'note4']])
            >>> table.get_image_notes(gid_list)
            ['note1', 'note2', 'note3', 'note4']
        """

        note_list = self.get(("note",), gid_list)
        return note_list

    def get_image_paths(self, gid_list):
        """
        Gets list of absolute paths corresponding to given gids.
        Gets single path if given one gid.

        Parameters:
            gid_list (list): list of gids

        Returns:
            gpath_list (list): list of corresponding absolute paths

        Doctest Command:
            python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE db/table.py

        Example:
            >>> table = ImageTable("test_data/", ["grevy's zebra"])
            >>> gid_list = table.add_image_data(['uuid', 'uri'],
            ... [['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', 'DSCN5265.JPG']])
            >>> table.get_image_paths(gid_list[0])
            'test_data/images/DSCN5265.JPG'
            >>> gid_list = table.add_image_data(['uuid', 'uri'],
            ... [['11ca4e9b-1de9-4ee5-8d90-615bf93077c8', 'DSCN5266.JPG'],
            ... ['6830cd43-ca00-28ee-8843-fe64d789d7f7', 'DSCN5267.JPG'],
            ... ['b54193f2-8507-8441-5bf3-cd9f0ed8a883', 'DSCN5268.JPG']])
            >>> table.get_image_paths(gid_list)
            ['test_data/images/DSCN5265.JPG', 'test_data/images/DSCN5266.JPG',
             'test_data/images/DSCN5267.JPG', 'test_data/images/DSCN5268.JPG']
        """

        uri_list = self.get_image_uris(gid_list)

        def islocal(uri):
            return not isabs(uri)

        gpath_list = []
        if not ut.isiterable(uri_list):
            if uri_list is None:
                gpath = None
            elif isabs(uri_list):
                gpath = uri_list
            else:
                assert islocal(uri_list)
                gpath = join(self.imgdir, uri_list)
            return gpath

        for uri in uri_list:
            if uri is None:
                gpath = None
            elif isabs(uri):
                gpath = uri
            else:
                assert islocal(uri)
                gpath = join(self.imgdir, uri)
            gpath_list.append(gpath)

        return gpath_list

    def get_image_location_codes(self, gid_list):
        """
        Gets list of location codes corresponding to given gids.
        Gets single location code if given one gid

        Parameters:
            gid_list (list): list of gids

        Returns:
            loc_list (list): list of corresponding location codes

        Doctest Command:
            python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE db/table.py

        Example:
            >>> table = ImageTable("test_data/", ["grevy's zebra"])
            >>> gid_list = table.add_image_data(['uuid', 'location_code'],
            ... [['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', 'GGR1']])
            >>> table.get_image_location_codes(gid_list[0])
            'GGR1'
            >>> gid_list = table.add_image_data(['uuid', 'location_code'],
            ... [['11ca4e9b-1de9-4ee5-8d90-615bf93077c8', 'GGR2'],
            ... ['6830cd43-ca00-28ee-8843-fe64d789d7f7', 'GGR3'],
            ... ['b54193f2-8507-8441-5bf3-cd9f0ed8a883', 'GGR4']])
            >>> table.get_image_location_codes(gid_list)
            ['GGR1', 'GGR2', 'GGR3', 'GGR4']
        """

        loc_list = self.get(("location_code",), gid_list)
        return loc_list

    def get_image_reviewed(self, gid_list):
        """
        Gets list of reviewed flags corresponding to given gids.
        Gets single reviewed flag if given one gid.

        Parameters:
            gid_list (list): list of gids

        Returns:
            reviewed_list (list): list of corresponding reviewed flags
            (true if all objects of interest (animals) have an ANNOTATION in the image)

        Doctest Command:
            python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE db/table.py

        Example:
            >>> table = ImageTable("test_data/", ["grevy's zebra"])
            >>> gid_list = table.add_image_data(['uuid', 'reviewed'],
            ... [['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', True]])
            >>> table.get_image_reviewed(gid_list[0])
            True
            >>> gid_list = table.add_image_data(['uuid', 'reviewed'],
            ... [['11ca4e9b-1de9-4ee5-8d90-615bf93077c8', False],
            ... ['6830cd43-ca00-28ee-8843-fe64d789d7f7', True],
            ... ['b54193f2-8507-8441-5bf3-cd9f0ed8a883', False]])
            >>> table.get_image_reviewed(gid_list)
            [True, False, True, False]
        """

        reviewed_list = self.get(("reviewed",), gid_list)
        return reviewed_list

    def get_image_exif_original(self, gid_list):
        """
        Gets list of dictionaries containing exif data corresponding to given gids.
        Gets single exif dictionary if given one gid.

        Parameters:
            gid_list (list): list of gids

        Returns:
            exif_dict_list (list): list of corresponding exif dictionaries

        Doctest Command:
            python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE db/table.py

        Example:
            >>> import exif
            >>> import os
            >>> import numpy
            >>> import shutil
            >>> from PIL import Image
            >>> db_path = "test_data/"
            >>> os.makedirs(db_path + "test_data")
            >>> for n in range(2):
            ...     a = numpy.random.rand(30,30,3) * 255
            ...     img = Image.fromarray(a.astype('uint8')).convert('RGB')
            ...     img.save(db_path + ("test_data/img%000d.jpg" % n))
            >>> gpath_list = [db_path + "test_data/img0.jpg", db_path + "test_data/img1.jpg"]
            >>> with open(gpath_list[0], 'rb') as img_file:
            ...     img = exif.Image(img_file)
            >>> img.gps_latitude = (1, 17, 30.786)
            >>> img.gps_latitude_ref = 'S'
            >>> img.gps_longitude = (36, 53, 53.4762)
            >>> img.gps_longitude_ref = 'E'
            >>> img.datetime = "2024:06:28 17:58:16"
            >>> img.orientation = 1
            >>> with open(gpath_list[0], 'wb') as img_file:
            ...     bytes = img_file.write(img.get_file())
            >>> table = ImageTable("test_data/test_dataset", ["grevy's zebra"])
            >>> gid_list = table.add_image_data(['uuid', 'uri_original'],
            ... [['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', gpath_list[0]]])
            >>> table.get_image_exif_original(gid_list[0])
            (1719611896, -1.291885, 36.89818783333333, 1, 30, 30)
            >>> gid_list = table.add_image_data(['uuid', 'uri_original'],
            ... [['11ca4e9b-1de9-4ee5-8d90-615bf93077c8', gpath_list[1]]])
            >>> table.get_image_exif_original(gid_list)
            [(1719611896, -1.291885, 36.89818783333333, 1, 30, 30), (-1, -1, -1, 0, 30, 30)]
            >>> shutil.rmtree(db_path + "test_data")
        """

        uri_list = self.get_image_uris_original(gid_list)
        exif_tup_list = []

        if not ut.isiterable(uri_list):
            uri_list = [uri_list]

        for uri in uri_list:
            pil_img = Image.open(uri, "r")
            exif_tup = get_exif(pil_img, uri)
            exif_tup_list.append(exif_tup)

        if len(exif_tup_list) == 1:
            return exif_tup_list[0]

        return exif_tup_list

    def set(self, colnames, gid_list, val_iter):
        """
        Sets values in image table corresponding to given column names and gids.

        Parameters:
            colnames (list): list of column names
            gid_list (list): list of gids
            val_iter (list): list of tuples, where each tuple corresponds to a gid
            and contains values for each column in colnames

        Doctest Command:
            python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE db/table.py

        Example:
            >>> table = ImageTable("test_data/", ["grevy's zebra"])
            >>> gid_list = table.add_image_data(['uuid', 'uri', 'original_name', 'ext'],
            ... [['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', '/data/GGR2020/QR100_A/Day1/DSCN5265.JPG', 'DSCN5265', '.JPG'],
            ... ['11ca4e9b-1de9-4ee5-8d90-615bf93077c8', '/data/GGR2020/QR100_A/Day1/DSCN5266.JPG', 'DSCN5266', '.JPG'],
            ... ['6830cd43-ca00-28ee-8843-fe64d789d7f7', '/data/GGR2020/QR100_A/Day1/DSCN5267.JPG', 'DSCN5267', '.JPG'],
            ... ['b54193f2-8507-8441-5bf3-cd9f0ed8a883', '/data/GGR2020/QR100_A/Day1/DSCN5268.JPG', 'DSCN5268', '.JPG']])
            >>> table.set(('uuid', 'uri', 'original_name', 'ext'), gid_list, [('uuid-set1', 'uri-set1', 'gname-set1', 'ext-set1'),
            ...                                                               ('uuid-set2', 'uri-set2', 'gname-set2', 'ext-set2'),
            ...                                                               ('uuid-set3', 'uri-set3', 'gname-set3', 'ext-set3'),
            ...                                                               ('uuid-set4', 'uri-set4', 'gname-set4', 'ext-set4')])
            >>> table.get(('uuid', 'uri', 'original_name', 'ext'), gid_list)
            [('uuid-set1', 'uri-set1', 'gname-set1', 'ext-set1'), ('uuid-set2', 'uri-set2', 'gname-set2', 'ext-set2'),
            ('uuid-set3', 'uri-set3', 'gname-set3', 'ext-set3'), ('uuid-set4', 'uri-set4', 'gname-set4', 'ext-set4')]
            >>> table.set(('original_name', 'ext'), gid_list, [('DSCN5265', '.JPG'), ('DSCN5266', '.JPG'),
            ...                                                ('DSCN5267', '.JPG'), ('DSCN5268', '.JPG')])
            >>> table.get(('original_name', 'ext'), gid_list)
            [('DSCN5265', '.JPG'), ('DSCN5266', '.JPG'), ('DSCN5267', '.JPG'), ('DSCN5268', '.JPG')]
            >>> table.set(('original_name',), gid_list, [('DSCN5260',), ('DSCN5261',), ('DSCN5262',), ('DSCN5263',)])
            >>> table.get(('original_name',), gid_list)
            ['DSCN5260', 'DSCN5261', 'DSCN5262', 'DSCN5263']
            >>> table.set(('original_name', 'ext'), gid_list[:2], [('gname-set1', 'ext-set1'), ('gname-set2', 'ext-set2')])
            >>> table.get(('original_name', 'ext'), gid_list[:2])
            [('gname-set1', 'ext-set1'), ('gname-set2', 'ext-set2')]
            >>> table.set(('original_name', 'ext'), [gid_list[0]], [('DSCN5265', '.JPG')])
            >>> table.get(('original_name', 'ext'), [gid_list[0]])
            [('DSCN5265', '.JPG')]
            >>> table.set(('original_name', 'ext'), gid_list[0], ('gname-set1', 'ext-set1'))
            >>> table.get(('original_name', 'ext'), gid_list[0])
            ('gname-set1', 'ext-set1')
            >>> table.set(('original_name',), gid_list[:2], [('DSCN5265',), ('DSCN5266',)])
            >>> table.get(('original_name',), gid_list[:2])
            ['DSCN5265', 'DSCN5266']
            >>> table.set(('original_name',), [gid_list[0]], [('gname-set1',)])
            >>> table.get(('original_name',), [gid_list[0]])
            ['gname-set1']
            >>> table.set(('original_name',), gid_list[0], ('DSCN5265',))
            >>> table.get(('original_name',), gid_list[0])
            'DSCN5265'
            >>> table.set(('original_name',), [], [])
        """

        if len(colnames) == 0 or (ut.isiterable(gid_list) and len(gid_list) == 0):
            return

        if not ut.isiterable(gid_list):
            gid_list = [gid_list]

        if type(val_iter) != list:
            val_iter = [val_iter]

        for col_idx in range(len(colnames)):
            colname = colnames[col_idx]
            for row_idx in range(len(gid_list)):
                gid = gid_list[row_idx]
                self.table[colname][gid - 1] = val_iter[row_idx][col_idx]

    def set_image_uris(self, gid_list, gpath_list):
        """
        Sets the image uris for images corresponding to given gids to a new local path.
        This is used when localizing or unlocalizing images.
        An absolute path is on this machine.
        A relative path is relative to the image cache on this machine.

        Parameters:
            gid_list (list): list of gids
            uri_list (list): list of new image uris

        Doctest Command:
            python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE db/table.py

        Example:
            >>> table = ImageTable("test_data/", ["grevy's zebra"])
            >>> gid_list = table.add_image_data(['uuid', 'uri'],
            ... [['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', '/data/GGR2020/QR100_A/Day1/DSCN5265.JPG']])
            >>> table.set_image_uris(gid_list[0], 'uri-set1')
            >>> table.get_image_uris(gid_list[0])
            'uri-set1'
            >>> gid_list = table.add_image_data(['uuid', 'uri'],
            ... [['11ca4e9b-1de9-4ee5-8d90-615bf93077c8', '/data/GGR2020/QR100_A/Day1/DSCN5266.JPG'],
            ... ['6830cd43-ca00-28ee-8843-fe64d789d7f7', '/data/GGR2020/QR100_A/Day1/DSCN5267.JPG'],
            ... ['b54193f2-8507-8441-5bf3-cd9f0ed8a883', '/data/GGR2020/QR100_A/Day1/DSCN5268.JPG']])
            >>> table.set_image_uris(gid_list, ['uri-set1', 'uri-set2', 'uri-set3', 'uri-set4'])
            >>> table.get_image_uris(gid_list)
            ['uri-set1', 'uri-set2', 'uri-set3', 'uri-set4']
        """

        if not ut.isiterable(gpath_list):
            gpath_list = [gpath_list]

        val_list = [(new_gpath,) for new_gpath in gpath_list]
        self.set(("uri",), gid_list, val_list)

    def set_image_unixtimes(self, gid_list, time_list):
        """
        Sets the unixtimes for images corresponding to given gids to new times.
        This is used when applying timedeltas to groups of images.

        Parameters:
            gid_list (list): list of gids
            time_list (list): list of new image unixtimes

        Doctest Command:
            python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE db/table.py

        Example:
            >>> table = ImageTable("test_data/", ["grevy's zebra"])
            >>> gid_list = table.add_image_data(['uuid', 'time_posix'],
            ... [['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', -1]])
            >>> table.set_image_unixtimes(gid_list[0], 1719611896)
            >>> table.get_image_unixtimes(gid_list[0])
            1719611896
            >>> gid_list = table.add_image_data(['uuid', 'time_posix'],
            ... [['11ca4e9b-1de9-4ee5-8d90-615bf93077c8', -1],
            ... ['6830cd43-ca00-28ee-8843-fe64d789d7f7', -1],
            ... ['b54193f2-8507-8441-5bf3-cd9f0ed8a883', -1]])
            >>> table.set_image_unixtimes(gid_list, [1719611896, 1719611897, 1719611898, 1719611899])
            >>> table.get_image_unixtimes(gid_list)
            [1719611896, 1719611897, 1719611898, 1719611899]
        """

        if not ut.isiterable(time_list):
            time_list = [time_list]

        val_list = [(new_time,) for new_time in time_list]
        self.set(("time_posix",), gid_list, val_list)

    def set_image_gps(self, gid_list, gps_list=None, lat_list=None, lon_list=None):
        """
        Sets the image latitude and longitude for images corresponding to given gids to
        a new latitude and longitude.

        Parameters:
            gid_list (list): list of gids
            gps_list (list): list of new image gps tuples
            lat_list (list): list of new image latitudes (optionally used in place of gps list)
            lon_list (list): list of new image longitudes (optionally used in place of gps list)

        Doctest Command:
            python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE db/table.py

        Example:
            >>> table = ImageTable("test_data/", ["grevy's zebra"])
            >>> gid_list = table.add_image_data(['uuid', 'gps_lat', 'gps_lon'],
            ... [['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', 0.291885, 36.89818783]])
            >>> table.set_image_gps(gid_list[0], (-1, -1))
            >>> table.get_image_gps(gid_list[0])
            (-1, -1)
            >>> table.set_image_gps([gid_list[0]], lat_list=0.291885, lon_list=36.89818783)
            >>> table.get_image_gps(gid_list[0])
            (0.291885, 36.89818783)
            >>> gid_list = table.add_image_data(['uuid', 'gps_lat', 'gps_lon'],
            ... [['11ca4e9b-1de9-4ee5-8d90-615bf93077c8', 1.291885, 37.89818783],
            ... ['6830cd43-ca00-28ee-8843-fe64d789d7f7', 2.291885, 38.89818783],
            ... ['b54193f2-8507-8441-5bf3-cd9f0ed8a883', 3.291885, 39.89818783]])
            >>> table.set_image_gps(gid_list, [(-1, -1), (-1, -1), (-1, -1), (-1, -1)])
            >>> table.get_image_gps(gid_list)
            [(-1, -1), (-1, -1), (-1, -1), (-1, -1)]
            >>> table.set_image_gps(gid_list, lat_list=[0.291885, 1.291885, 2.291885, 3.291885], lon_list=[36.89818783, 37.89818783, 38.89818783, 39.89818783])
            >>> table.get_image_gps(gid_list)
            [(0.291885, 36.89818783), (1.291885, 37.89818783), (2.291885, 38.89818783), (3.291885, 39.89818783)]
        """

        if gps_list is not None:
            assert lat_list is None
            assert lon_list is None
            if type(gps_list) == tuple:
                gps_list = [gps_list]

            lat_list = [tup[0] for tup in gps_list]
            lon_list = [tup[1] for tup in gps_list]

        if not ut.isiterable(lat_list):
            lat_list = [lat_list]

        if not ut.isiterable(lon_list):
            lon_list = [lon_list]

        val_list = [(lat, lon) for lat, lon in zip(lat_list, lon_list)]
        self.set(("gps_lat", "gps_lon"), gid_list, val_list)

    def set_image_orientations(self, gid_list, orient_list, clean_derivatives=True):
        """
        Sets the image orientations for images corresponding to given gids to a new orientation.

        Parameters:
            gid_list (list): list of gids
            orient_list (list): list of new image uris

        Doctest Command:
            python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE db/table.py

        Example:
            >>> table = ImageTable("test_data/", ["grevy's zebra"])
            >>> gid_list = table.add_image_data(['uuid', 'orientation'],
            ... [['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', 0]])
            >>> table.set_image_orientations(gid_list[0], 1)
            >>> table.get_image_orientations(gid_list[0])
            1
            >>> gid_list = table.add_image_data(['uuid', 'orientation'],
            ... [['11ca4e9b-1de9-4ee5-8d90-615bf93077c8', 2],
            ... ['6830cd43-ca00-28ee-8843-fe64d789d7f7', 3],
            ... ['b54193f2-8507-8441-5bf3-cd9f0ed8a883', 4]])
            >>> table.set_image_orientations(gid_list, [0, 0, 0, 0])
            >>> table.get_image_orientations(gid_list)
            [0, 0, 0, 0]
        """

        if not ut.isiterable(orient_list):
            orient_list = [orient_list]

        val_list = [(orientation,) for orientation in orient_list]
        self.set(("orientation",), gid_list, val_list)

        # if clean_derivatives:
        #     # Delete image's thumbs
        #     ibs.depc_image.notify_root_changed(gid_list, 'image_orientation')
        #     ibs.delete_image_thumbs(gid_list)

        #     # Delete annotation's thumbs
        #     aid_list = ut.flatten(ibs.get_image_aids(gid_list))
        #     ibs.delete_annot_chips(aid_list)
        #     ibs.delete_annot_imgthumbs(aid_list)
        #     gid_list = list(set(ibs.get_annot_gids(aid_list)))
        #     config2_ = {'thumbsize': 221}
        #     ibs.delete_image_thumbs(gid_list, quiet=True, **config2_)

    def set_image_notes(self, gid_list, note_list):
        """
        Sets the image notes for images corresponding to given gids to a new note.

        Parameters:
            gid_list (list): list of gids
            note_list (list): list of new image notes

        Doctest Command:
            python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE db/table.py

        Example:
            >>> table = ImageTable("test_data/", ["grevy's zebra"])
            >>> gid_list = table.add_image_data(['uuid', 'note'],
            ... [['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', 'note1']])
            >>> table.set_image_notes(gid_list[0], 'note-set1')
            >>> table.get_image_notes(gid_list[0])
            'note-set1'
            >>> gid_list = table.add_image_data(['uuid', 'note'],
            ... [['11ca4e9b-1de9-4ee5-8d90-615bf93077c8', 'note2'],
            ... ['6830cd43-ca00-28ee-8843-fe64d789d7f7', 'note3'],
            ... ['b54193f2-8507-8441-5bf3-cd9f0ed8a883', 'note4']])
            >>> table.set_image_notes(gid_list, ['note-set1', 'note-set2', 'note-set3', 'note-set4'])
            >>> table.get_image_notes(gid_list)
            ['note-set1', 'note-set2', 'note-set3', 'note-set4']
        """

        if not ut.isiterable(note_list):
            note_list = [note_list]

        val_list = [(new_note,) for new_note in note_list]
        self.set(("note",), gid_list, val_list)

    def set_image_location_codes(self, gid_list, loc_list):
        """
        Sets the image location codes for images corresponding to given gids to a new location code.

        Parameters:
            gid_list (list): list of gids
            loc_list (list): list of new image location codes

        Doctest Command:
            python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE db/table.py

        Example:
            >>> table = ImageTable("test_data/", ["grevy's zebra"])
            >>> gid_list = table.add_image_data(['uuid', 'location_code'],
            ... [['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', 'GGR1']])
            >>> table.set_image_location_codes(gid_list[0], 'GGR-set1')
            >>> table.get_image_location_codes(gid_list[0])
            'GGR-set1'
            >>> gid_list = table.add_image_data(['uuid', 'location_code'],
            ... [['11ca4e9b-1de9-4ee5-8d90-615bf93077c8', 'GGR2'],
            ... ['6830cd43-ca00-28ee-8843-fe64d789d7f7', 'GGR3'],
            ... ['b54193f2-8507-8441-5bf3-cd9f0ed8a883', 'GGR4']])
            >>> table.set_image_location_codes(gid_list, ['GGR-set1', 'GGR-set2', 'GGR-set3', 'GGR-set4'])
            >>> table.get_image_location_codes(gid_list)
            ['GGR-set1', 'GGR-set2', 'GGR-set3', 'GGR-set4']
        """

        if not ut.isiterable(loc_list):
            loc_list = [loc_list]

        val_list = [(new_loc,) for new_loc in loc_list]
        self.set(("location_code",), gid_list, val_list)

    def set_image_reviewed(self, gid_list, reviewed_list):
        """
        Sets the image reviewed flag for images corresponding to given gids to a new flag.

        Parameters:
            gid_list (list): list of gids
            reviewed_list (list): list of new image reviewed flag

        Doctest Command:
            python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE db/table.py

        Example:
            >>> table = ImageTable("test_data/", ["grevy's zebra"])
            >>> gid_list = table.add_image_data(['uuid', 'reviewed'],
            ... [['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', True]])
            >>> table.set_image_reviewed(gid_list[0], False)
            >>> table.get_image_reviewed(gid_list[0])
            False
            >>> gid_list = table.add_image_data(['uuid', 'reviewed'],
            ... [['11ca4e9b-1de9-4ee5-8d90-615bf93077c8', False],
            ... ['6830cd43-ca00-28ee-8843-fe64d789d7f7', False],
            ... ['b54193f2-8507-8441-5bf3-cd9f0ed8a883', True]])
            >>> table.set_image_reviewed(gid_list, [True, True, True, False])
            >>> table.get_image_reviewed(gid_list)
            [True, True, True, False]
        """

        if not ut.isiterable(reviewed_list):
            reviewed_list = [reviewed_list]

        val_list = [(new_flag,) for new_flag in reviewed_list]
        self.set(("reviewed",), gid_list, val_list)

    def flag_poly_contains(self, gid_list, poly):
        """
        Gets list of flags corresponding to whether images with given gids were taken within the given geo-fence.
        Gets single flag if given one gid. (points on polygon boundary are flagged as false)

        Parameters:
            gid_list (list): list of gids
            poly (list / Polygon): list of gps tuples or shapely Polygon object representing a geo-fence

        Returns:
            flag_list (list): list of corresponding flags
            (true if image was taken within the geo-fence)

        Doctest Command:
            python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE db/table.py

        Example:
            >>> table = ImageTable("test_data/", ["grevy's zebra"])
            >>> coordinates = [(51.509764, -0.189529), (51.512819, -0.159570), (51.504451, -0.152143),
            ...                          (51.502639, -0.187329), (51.509764, -0.189529)]
            >>> gid_list = table.add_image_data(['uuid', 'gps_lat', 'gps_lon'],
            ... [['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', 51.505278, -0.159479],
            ... ['11ca4e9b-1de9-4ee5-8d90-615bf93077c8', 51.496443, -0.176218]])
            >>> table.flag_poly_contains(gid_list, coordinates)
            [True, False]
            >>> table.flag_poly_contains(gid_list, Polygon(coordinates))
            [True, False]
            >>> table.flag_poly_contains(gid_list[0], coordinates)
            True
            >>> table.flag_poly_contains([], coordinates)
            []
            >>> coordinates = [(51.509764, -0.189529), (51.512819, -0.159570),
            ...                (51.504451, -0.152143), (51.502639, -0.187329)]
            >>> table.flag_poly_contains(gid_list, coordinates)
            [True, False]
            >>> table.set_image_gps(gid_list[0], (51.509764, -0.189529))
            >>> table.flag_poly_contains(gid_list[0], coordinates)
            False
        """

        if type(poly) == list:
            poly = Polygon(poly)

        gps_list = self.get_image_gps(gid_list)
        if type(gps_list) != list:
            gps_list = [gps_list]

        flag_list = []

        for gps in gps_list:
            point = Point(gps)
            flag_list.append(poly.contains(point))

        if not ut.isiterable(gid_list):
            return flag_list[0]

        return flag_list

    def poly_contains(self, gid_list, poly):
        """
        Gets list of gids corresponding to images taken within the given geo-fence.
        (points on polygon boundary are flagged as false)

        Parameters:
            gid_list (list): list of gids
            poly (list / Polygon): list of gps tuples or shapely Polygon object representing a geo-fence

        Returns:
            gid_list_compressed (list): list of gids of images taken within the given geo-fence

        Doctest Command:
            python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE db/table.py

        Example:
            >>> table = ImageTable("test_data/", ["grevy's zebra"])
            >>> coordinates = [(51.509764, -0.189529), (51.512819, -0.159570), (51.504451, -0.152143),
            ...                          (51.502639, -0.187329), (51.509764, -0.189529)]
            >>> gid_list = table.add_image_data(['uuid', 'gps_lat', 'gps_lon'],
            ... [['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', 51.505278, -0.159479],
            ... ['11ca4e9b-1de9-4ee5-8d90-615bf93077c8', 51.496443, -0.176218]])
            >>> table.poly_contains(gid_list, coordinates)
            [1]
            >>> table.poly_contains(gid_list, Polygon(coordinates))
            [1]
            >>> table.poly_contains(gid_list[0], coordinates)
            [1]
            >>> table.poly_contains([], coordinates)
            []
            >>> gid_list = table.add_image_data(['uuid', 'gps_lat', 'gps_lon'],
            ... [['b54193f2-8507-8441-5bf3-cd9f0ed8a883', 51.506339, -0.168462]])
            >>> table.poly_contains(gid_list, coordinates)
            [1, 3]
            >>> coordinates = [(51.509764, -0.189529), (51.512819, -0.159570),
            ...                (51.504451, -0.152143), (51.502639, -0.187329)]
            >>> table.poly_contains(gid_list, coordinates)
            [1, 3]
            >>> table.set_image_gps(gid_list[0], (51.509764, -0.189529))
            >>> table.poly_contains(gid_list[0], coordinates)
            []
        """

        if not ut.isiterable(gid_list):
            gid_list = [gid_list]

        flag_list = self.flag_poly_contains(gid_list, poly)

        gid_list_compressed = ut.list_compress(gid_list, flag_list)

        return gid_list_compressed

    def polys_contain(self, gid_list, poly_list):
        """
        Gets list of lists of gids corresponding to images taken within the given geo-fences.
        (points on polygon boundaries are flagged as false)

        Parameters:
            gid_list (list): list of gids
            poly_list (list / Polygon): list of shapely Polygon objects or lists of gps tuples representing a geo-fence

        Returns:
            gid_lists_compressed (list): list of lists of gids of images taken within corresponding provided geo-fences

        Doctest Command:
            python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE db/table.py

        Example:
            >>> table = ImageTable("test_data/", ["grevy's zebra"])
            >>> coordinates = [(51.509764, -0.189529), (51.512819, -0.159570), (51.504451, -0.152143),
            ...                          (51.502639, -0.187329), (51.509764, -0.189529)]
            >>> gid_list = table.add_image_data(['uuid', 'gps_lat', 'gps_lon'],
            ... [['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', 51.505278, -0.159479],
            ... ['11ca4e9b-1de9-4ee5-8d90-615bf93077c8', 51.496443, -0.176218]])
            >>> table.polys_contain(gid_list, coordinates)
            [1]
            >>> gid_list = table.add_image_data(['uuid', 'gps_lat', 'gps_lon'],
            ... [['b54193f2-8507-8441-5bf3-cd9f0ed8a883', 51.506339, -0.168462]])
            >>> coordinates2 = [(51.309764, -0.189529), (51.304451, -0.162143),
            ...                (51.522819, -0.159570), (51.522639, -0.187329)]
            >>> table.polys_contain(gid_list[0], [coordinates, coordinates2])
            [[1], []]
            >>> table.polys_contain([], [coordinates, coordinates2])
            [[], []]
            >>> table.polys_contain(gid_list, [coordinates, coordinates2])
            [[1, 3], [2, 3]]
        """

        if not ut.isiterable(poly_list) or type(poly_list[0]) == tuple:
            return self.poly_contains(gid_list, poly_list)

        gid_lists_compressed = []

        for poly in poly_list:
            gid_lists_compressed.append(self.poly_contains(gid_list, poly))

        return gid_lists_compressed
