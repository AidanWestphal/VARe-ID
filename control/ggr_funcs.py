import cv2
import geopandas # type: ignore
import json
from qreader import QReader
from shapely import Polygon


def extrapolate_ggr_gps(imgtable, doctest_mode=False):
    """
    Extrapolates GPS data for images taken by cameras without GPS data using
    cameras with GPS within the same cars. Synchronizes image unixtimes within each car
    based on times QR code images were taken on each camera.

    Parameters:
        imgtable (ImageTable): table containing image metadata
        doctest_mode (bool): if true, replaces carriage returns with newlines
    
    Returns:
        skipped_gid_list (list): list of gids for images that GPS data could not be extrapolated for

    Doctest Command:
        python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE control/ggr_funcs.py
    
    Example:
        >>> import os
        >>> import numpy
        >>> import shutil
        >>> from PIL import Image
        >>> from db.table import ImageTable
        >>> db_path = "/mnt/c/Users/Julian/Research/Pipeline/data/"
        >>> os.makedirs(db_path + "test_data/QR100_A/Day1")
        >>> os.makedirs(db_path + "test_data/QR100_B/Day1")
        >>> for n in range(2):
        ...     a = numpy.random.rand(30,30,3) * 255
        ...     img = Image.fromarray(a.astype('uint8')).convert('RGB')
        ...     img.save(db_path + ("test_data/QR100_A/Day1/img%000d.jpg" % n))
        >>> for n in range(2, 4):
        ...     a = numpy.random.rand(30,30,3) * 255
        ...     img = Image.fromarray(a.astype('uint8')).convert('RGB')
        ...     img.save(db_path + ("test_data/QR100_B/Day1/img%000d.jpg" % n))
        >>> gpath_list = [db_path + "test_data/QR100_A/Day1/img0.jpg", db_path + "test_data/QR100_A/Day1/img1.jpg",
        ...               db_path + "test_data/QR100_B/Day1/img2.jpg", db_path + "test_data/QR100_B/Day1/img3.jpg"]
        >>> table = ImageTable("/mnt/c/Users/Julian/Research/Pipeline/data/test_dataset", ["grevy's zebra"])
        >>> gid_list = table.add_image_data(['uuid', 'uri_original', 'time_posix', 'gps_lat', 'gps_lon', 'note'],
        ... [['11ca4e9b-1de9-4ee5-8d90-615bf93077c8', gpath_list[0], 1000000, 0.5, 30.0, ''],
        ... ['6830cd43-ca00-28ee-8843-fe64d789d7f7', gpath_list[1], 2000000, 0.7, 33.0, ''],
        ... ['0fdef8e8-cec0-b460-bac2-6ee3e39f0798', gpath_list[2], 1500000, -1, -1, ''],
        ... ['b54193f2-8507-8441-5bf3-cd9f0ed8a883', gpath_list[3], 2500000, -1, -1, '']])
        >>> skipped_gid_list = extrapolate_ggr_gps(table, doctest_mode=True)
        [extrapolate_ggr_gps] 2/4 total images missing GPS data
        [extrapolate_ggr_gps] gathering image data [1/4]
        [extrapolate_ggr_gps] gathering image data [2/4]
        [extrapolate_ggr_gps] gathering image data [3/4]
        [extrapolate_ggr_gps] gathering image data [4/4]
        [extrapolate_ggr_gps] setting notes
        [extrapolate_ggr_gps] getting QR code images
        [extrapolate_ggr_gps] unable to locate QR code: {Car: QR100; Camera: A; Day: 1}
            ...skipping unixtime synchronization
        [extrapolate_ggr_gps] unable to locate QR code: {Car: QR100; Camera: B; Day: 1}
            ...skipping unixtime synchronization
        [extrapolate_ggr_gps] set camera A as GPS reference for car QR100 on day 1
        [extrapolate_ggr_gps] set GPS data for 2 / 2 images with missing GPS entries
        >>> shutil.rmtree(db_path + "test_data")
    """

    gid_list = imgtable.get_all_gids() # A list of image gids
    uri_list = imgtable.get_image_uris_original(gid_list) # A list of original image paths
    gps_list = imgtable.get_image_gps(gid_list)
    ggr_hierarchy = [] # A list of days, each a dictionary containing cars, each containing 3 aligned lists 
    # containing a list of image gids, QR code image gid, and QR code image URL for each camera
    car_list = [] # A list containing a car for each image
    unskipped_gid_list = []
    uri_lookup = dict() # A dictionary mapping gids to uris

    missing_gid_list = []
    for i in range(len(gid_list)):
        if gps_list[i] == (-1, -1):
            missing_gid_list.append(gid_list[i])

    print(f"[extrapolate_ggr_gps] {len(missing_gid_list)}/{len(gid_list)} total images missing GPS data")

    # Gather metadata from image hierarchy, store in ggr_hierarchy
    # Prepare additional lists for car and car/day imagesets
    for i in range(len(gid_list)):
        # Identify car, camera, & day from image hierarchy
        tokens = uri_list[i].split('/')
        cam_str, day_str = "", ""
        for token in tokens:
            if "QR" in token:
                cam_str = token
            elif "Day" in token:
                day_str = token

        if cam_str == "" or day_str == "":
            print(f"[extrapolate_ggr_gps] detected improper GGR directory format: {uri_list[i]}. Skipping image...")
            continue

        car = cam_str[:cam_str.find('_')]
        cam_idx = ord(cam_str[-1]) - 65

        # Handle doubly nested images in case of flaw in image hierarchy
        while "Day" not in day_str and "Other" not in day_str and str_day_idx < len(tokens) - 1:
            str_day_idx += 1
            day_str = tokens[str_day_idx]

        # Add gid to ggr_hierarchy, handle data for misordered or missing cars & cameras
        if day_str[-1].isdigit():
            day_idx = int(day_str[-1]) - 1
            while day_idx >= len(ggr_hierarchy):
                ggr_hierarchy.append(dict())

            if car not in ggr_hierarchy[day_idx].keys():
                ggr_hierarchy[day_idx][car] = [[], [], []]

            car_data = ggr_hierarchy[day_idx][car]
            while cam_idx >= len(car_data[0]):
                car_data[0].append([])
                car_data[1].append(0)
                car_data[2].append("")

            car_data[0][cam_idx].append(gid_list[i])

        # Track car for each gid, imagesets the image is in, and images within each imageset
        car_list.append(car)
        unskipped_gid_list.append(gid_list[i])

        if doctest_mode:
            print('[extrapolate_ggr_gps] gathering image data [%d/%d]\n' % (i + 1, len(gid_list)), end="")
        else:
            print('[extrapolate_ggr_gps] gathering image data [%d/%d]\r' % (i + 1, len(gid_list)), end="")

    print('\n[extrapolate_ggr_gps] setting notes')
    imgtable.set_image_notes(unskipped_gid_list, car_list)

    # Locate QR code images for each camera/day
    print('[extrapolate_ggr_gps] getting QR code images')
    qreader = QReader()
    for i in range(len(gid_list)):
        uri_lookup[gid_list[i]] = uri_list[i]

    # Iterate over each camera
    for day_idx in range(len(ggr_hierarchy)):
        day = ggr_hierarchy[day_idx]
        for car in day.keys():
            car_data = day[car]
            for cam_idx in range(len(car_data[0])):
                # Test camera's images until QR code is found
                for gid_idx in range(len(car_data[0][cam_idx])):
                    qr_image_gid = car_data[0][cam_idx][gid_idx]
                    image = cv2.cvtColor(cv2.imread(uri_lookup[qr_image_gid]), cv2.COLOR_BGR2RGB)
                    data = qreader.detect_and_decode(image)

                    # If desired QR code is found, store image gid and QR code URL in ggr_hierarchy
                    if len(data) != 0:
                        qr_image_url = data[0]

                        if not qr_image_url or "http" not in qr_image_url:
                            continue
                        
                        car_data[1][cam_idx] = qr_image_gid
                        car_data[2][cam_idx] = qr_image_url
                        qr_image_name = imgtable.get_image_gnames([qr_image_gid])[0]
                        print('[extrapolate_ggr_gps] found QR code: {Car: %s; Camera: %s; Day: %d; Image: %s; URL: %s}' 
                                    % (car, chr(cam_idx + 65), day_idx + 1, qr_image_name, qr_image_url,))
                        break
                
                if car_data[1][cam_idx] == 0:
                    print('[extrapolate_ggr_gps] unable to locate QR code: {Car: %s; Camera: %s; Day: %d}' 
                                    % (car, chr(cam_idx + 65), day_idx + 1,))
                    print("\t...skipping unixtime synchronization")

    # Synchronize unixtimes & calculate missing GPS data
    for day_idx in range(len(ggr_hierarchy)):
        for car in ggr_hierarchy[day_idx].keys():
            car_data = ggr_hierarchy[day_idx][car]
            imgsets = car_data[0]
            qr_gids = car_data[1]
            qr_times = imgtable.get_image_unixtimes(qr_gids)
            # Calculate timedeltas for each camera relative to first camera based on QR code images (taken simultaneously)
            base_times = [qr_times[0]] * len(qr_times)
            timedeltas = [time - base_time for time, base_time in zip(qr_times, base_times)]
            gps_cam_idx = -1

            # Locate camera with GPS data in each car
            for cam_idx in range(len(imgsets)):
                gids = imgsets[cam_idx]
                
                # Skip entries for missing cars
                if not gids:
                    continue

                # Set timedeltas for each image in the camera
                if qr_gids[cam_idx] != 0:
                    unixtimes = imgtable.get_image_unixtimes(gids)
                    imgtable.set_image_unixtimes(gids, [t - timedeltas[cam_idx] for t in unixtimes])
                    print('[extrapolate_ggr_gps] synchronized unixtimes: {Car: %s; Camera: %s; Day: %d}' 
                                    % (car, chr(cam_idx + 65), day_idx + 1,))


                # Mark camera as GPS camera if any of its images have GPS
                gps_list = imgtable.get_image_gps(gids)
                for i in range(len(gps_list)):
                    if gps_list[i] != (-1, -1):
                        gps_cam_idx = cam_idx
                        break

                # Extrapolate for missing GPS data in GPS camera, if any
                if cam_idx == gps_cam_idx:
                    # Sort images by time
                    times = imgtable.get_image_unixtimes(gids)
                    sorted_lists = sorted(zip(times, gids))
                    times_sorted, gids_sorted = zip(*sorted_lists)

                    
                    for i in range(len(gids_sorted)):
                        # Get GPS data from closest image in time with GPS data
                        if imgtable.get_image_gps(gids_sorted[i]) == (-1, -1):
                            # Search for closest past image with GPS data
                            j = i
                            while (j > 0 and imgtable.get_image_gps(gids_sorted[j]) == (-1, -1)):
                                j -= 1

                            # Search for closest future image with GPS data
                            k = i
                            while ((k < len(gids_sorted) - 1) and imgtable.get_image_gps(gids_sorted[k]) == (-1, -1)):
                                k += 1

                            # Set GPS data to closest from past or future
                            nearest_gps = imgtable.get_image_gps(gids_sorted[j])
                            if abs(times_sorted[j] - times_sorted[i]) > abs(times_sorted[k] - times_sorted[i]) or nearest_gps == (-1, -1):
                                nearest_gps = imgtable.get_image_gps(gids_sorted[k])
                            imgtable.set_image_gps([gids_sorted[i]], [nearest_gps])
                    
                    print(f'[extrapolate_ggr_gps] set camera {chr(cam_idx + 65)} as GPS reference for car {car} on day {day_idx + 1}')
                    break
                            
            # Extrapolate for missing GPS data in other cameras
            for cam_idx in range(len(imgsets)):
                # Skip GPS camera and missing cameras
                if cam_idx == gps_cam_idx or not imgsets[cam_idx]:
                    continue

                # Sort images for GPS camera (A) and current camera (B) by adjusted time
                gids_A = imgsets[gps_cam_idx]
                gids_B = imgsets[cam_idx]
                times = imgtable.get_image_unixtimes(gids_A)
                sorted_lists = sorted(zip(times, gids_A))
                times_sorted_A, gids_sorted_A = zip(*sorted_lists)
                gps_sorted_A = imgtable.get_image_gps(gids_sorted_A)
                times = imgtable.get_image_unixtimes(gids_B)
                sorted_lists = sorted(zip(times, gids_B))
                times_sorted_B, gids_sorted_B = zip(*sorted_lists)

                # Merge GPS data from GPS camera images and current camera images
                A_idx = 0
                B_idx = 0
                gps_sorted_B = []
                while B_idx < len(times_sorted_B):
                    if times_sorted_A <= times_sorted_B or A_idx >= len(times_sorted_A) - 1:
                        gps_sorted_B.append(gps_sorted_A[A_idx])
                        B_idx += 1
                    else:
                        A_idx += 1

                imgtable.set_image_gps(gids_sorted_B, gps_sorted_B)

    # Locate all images for which GPS data could not be extrapolated
    gps_list = imgtable.get_image_gps(gid_list)
    uri_list = imgtable.get_image_uris_original(gid_list)
    skipped_gid_list = []
    for i in range(len(gid_list)):
        if gps_list[i] == (-1, -1):
            skipped_gid_list.append(gid_list[i])

    print('[extrapolate_ggr_gps] set GPS data for %d / %d images with missing GPS entries' % (len(missing_gid_list) - len(skipped_gid_list), len(missing_gid_list)))
    if (skipped_gid_list):
        print('[extrapolate_ggr_gps] unable to extrapolate GPS data for %d images with the following gids:' % (len(skipped_gid_list)))
        print(f"\t{skipped_gid_list}")

    return skipped_gid_list


def get_ggr_polygons(filepath="ggr_counties.json", c_or_lt = 0, poly_obj=True):
    """
    Import GGR county/land holding polygons from 'ggr_counties.json' or another provided file.
    Converts coordinate lists to shapely Polygon objects if poly_obj is True.

    Parameters:
        filepath (str): path to json with GGR county geo-fence data
        c_or_lt (int): Determines whether to import using county or land tenure format (0 for county, 1 for land tenure)
        poly_obj (bool): If true, converts coordinate lists to shapely Polygon objects
    
    Returns:
        poly_dict (dict): Dictionary mapping county/land holding names to coordinate lists or polygons

    Doctest Command:
        python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE control/ggr_funcs.py
    
    Example:
        >>> poly_dict = get_ggr_polygons(poly_obj=False)
        [ggr_funcs.get_ggr_polygons] Imported polygons for 47 counties from json...
        >>> print(len(poly_dict["Turkana"][0]))
        5280
        >>> print(poly_dict["Turkana"][0][:10])
        [(5.344485759410418, 35.795925139933445), (5.344676018183861, 35.796592712084134), 
        (5.345106125247639, 35.79743194592771), (5.345294952300378, 35.79821014368338), 
        (5.345104217785604, 35.79901885993263), (5.344961642865712, 35.80001831069666), 
        (5.344852924722886, 35.801631927371716), (5.345104217785604, 35.802825927574645), 
        (5.345675468048569, 35.80353927611685), (5.346293926423755, 35.80411148056055)]
        >>> poly_dict = get_ggr_polygons()
        [ggr_funcs.get_ggr_polygons] Imported polygons for 47 counties from json...
        >>> print(poly_dict["Turkana"])
        [<POLYGON ((5.344 35.796, 5.345 35.797, 5.345 35.797, 5.345 35.798, 5.345 35....>]
        >>> print(len(poly_dict["Lamu"]))
        67
    """

    with open(filepath, 'r') as infile:
        in_dict = json.load(infile)
    
    poly_dict = dict()
    for county in in_dict["features"]:
        if not c_or_lt:
            name = county["properties"]["COUNTY"]
        else:
            name = county["properties"]["Name"]

        if county["geometry"]["type"] == "Polygon":
            coord_list = [(point[0], point[1]) for point in county["geometry"]["coordinates"][0]]

            if poly_obj:
                poly_dict[name] = [Polygon(coord_list)]
            else:
                poly_dict[name] = [coord_list]
        elif county["geometry"]["type"] == "MultiPolygon":
            poly_dict[name] = []
            for polygon in county["geometry"]["coordinates"]:
                coord_list = [(point[0], point[1]) for point in polygon[0]]

                if poly_obj:
                    poly_dict[name].append(Polygon(coord_list))
                else:
                    poly_dict[name].append(coord_list)
    
    land_type = "land tenures" if c_or_lt else "counties"
    print(f"[ggr_funcs.get_ggr_polygons] Imported polygons for {len(poly_dict)} {land_type} from json...")
    
    return poly_dict


def fix_json_lat_lon(filepath):
    """
    Switches latitude and longitude values in json geofence file for database compatibility.
    Removes excess gps coordinate tuple values when necessary.

    Parameters:
        filepath (str): path to json with GGR county or land holding geo-fence data

    Doctest Command:
        python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE control/ggr_funcs.py
    """

    with open(filepath, 'r') as infile:
        in_dict = json.load(infile)
    
    for county in in_dict["features"]:
        if county["geometry"]["type"] == "Polygon":
            for coord_list in county["geometry"]["coordinates"][0]:
                if len(coord_list) > 2:
                    coord_list.pop(2)

                tmp_val = coord_list[0]
                coord_list[0] = coord_list[1]
                coord_list[1] = tmp_val
            
        elif county["geometry"]["type"] == "MultiPolygon":
            for polygon in county["geometry"]["coordinates"]:
                for coord_list in polygon[0]:
                    if len(coord_list) > 2:
                        coord_list.pop(2)

                    tmp_val = coord_list[0]
                    coord_list[0] = coord_list[1]
                    coord_list[1] = tmp_val

    with open(filepath, 'w') as outfile:
        json.dump(in_dict, outfile)


def convert_geofence_to_json(filepath, outpath):
    """
    Converts kml or shp gps coordinate file to json gps coordinate file.
    If converting from shp, ensure .dbf and .shx files exist under the same name in the same directory.
    GGR county data can be found here: https://wildbookiarepository.azureedge.net/data/kenyan_counties_boundary_gps_coordinates.zip
    GGR land tenure data can be found here: https://wildbookiarepository.azureedge.net/data/kenyan_land_tenures_boundary_gps_coordinates.zip

    Parameters:
        filepath (str): path to shp or kml gps coordinate file
        outpath (str): path to output json file

    Doctest Command:
        python -W "ignore" -m doctest -o NORMALIZE_WHITESPACE control/ggr_funcs.py

    Example:
        >>> import utils as ut
        >>> lt_path = '/mnt/c/Users/Julian/Downloads/kenyan_land_tenures/'
        >>> db_path = '/mnt/c/Users/Julian/Research/Pipeline/GGR-Zebra-ReID/'
        >>> convert_geofence_to_json(lt_path + 'LandTenure.shp', db_path + 'ggr_landtenures_test.json')
        >>> poly_dict = get_ggr_polygons(filepath='ggr_landtenures_test.json', c_or_lt=1)
        [ggr_funcs.get_ggr_polygons] Imported polygons for 64 land tenures from json...
        >>> print(poly_dict["Enasoit"])
        [<POLYGON ((0.288 37.09, 0.27 37.095, 0.264 37.097, 0.248 37.088, 0.222 37.07...>]
        >>> convert_geofence_to_json(lt_path + 'LandTenures.kml', db_path + 'ggr_landtenures_test.json')
        >>> poly_dict = get_ggr_polygons(filepath='ggr_landtenures_test.json', c_or_lt=1)
        [ggr_funcs.get_ggr_polygons] Imported polygons for 64 land tenures from json...
        >>> print(poly_dict["Enasoit"])
        [<POLYGON ((0.288 37.09, 0.263 37.077, 0.259 37.041, 0.251 37.049, 0.248 37.0...>]
        >>> success = ut.remove_file(db_path + 'ggr_landtenures_test.json')
        [utils.remove_file] Finished deleting path='/mnt/c/Users/Julian/Research/Pipeline/GGR-Zebra-ReID/ggr_landtenures_test.json'
    """

    myshpfile = geopandas.read_file(filepath)
    myshpfile.to_file(outpath, driver='GeoJSON')
    fix_json_lat_lon(outpath)