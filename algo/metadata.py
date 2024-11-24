# import exifread
#
# def extract_metadata(filename):
#     with open(filename, 'rb') as f:
#         tags = exifread.process_file(f)
#
#     for tag in tags.keys():
#         print(f"{tag}: {tags[tag]}")
#
# if __name__ == '__main__':
#     filename = "/Users/jaeseok/Developer/GGR/Imageomics-Species-ReID/GGR2020_subset/QR21_C/Day1/IMG_6461.JPG"  # Replace with your image filename
#     extract_metadata(filename)
#
#
#
# from PIL import Image
# from PIL.ExifTags import TAGS
#
# # Load the image file
# image_path = "/Users/jaeseok/Developer/GGR/Imageomics-Species-ReID/GGR2020_subset/QR21_C/Day1/IMG_6461.JPG"
# image = Image.open(image_path)
#
# # Extract EXIF data
# exif_data = image.getexif()
#
# # Convert EXIF data to readable tags and check for date-related fields
# exif_info = {}
# for tag, value in exif_data.items():
#     tag_name = TAGS.get(tag, tag)
#     exif_info[tag_name] = value
#
# # Display all EXIF data
# exif_info
#
# print(exif_info)


import os
from datetime import datetime

# Path to the image file
image_path = "/Users/jaeseok/Developer/GGR/Imageomics-Species-ReID/GGR2020_subset/QR21_C/Day1/IMG_6461.JPG"

# Get the creation and modification times from the filesystem
creation_time = os.path.getctime(image_path)
modification_time = os.path.getmtime(image_path)

# Convert to human-readable format
creation_date = datetime.fromtimestamp(creation_time).strftime('%Y/%m/%d %H:%M:%S')
modification_date = datetime.fromtimestamp(modification_time).strftime('%Y/%m/%d %H:%M:%S')

creation_date, modification_date

print(creation_date, modification_date)
