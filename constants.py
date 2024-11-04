import six
from PIL.ExifTags import TAGS

EXIF_TIME = 'DateTime'
EXIF_LAT = 'GPSLatitude'
EXIF_LAT_REF = 'GPSLatitudeRef'
EXIF_LON = 'GPSLongitude'
EXIF_LON_REF = 'GPSLongitudeRef'
EXIF_ORIENT = 'Orientation'
EXIF_HEIGHT = 'ExifImageHeight'
EXIF_WIDTH = 'ExifImageWidth'
EXIF_TAG_TO_TAGID = {val: key for (key, val) in six.iteritems(TAGS)}
ORIENTATION_CODE = EXIF_TAG_TO_TAGID['Orientation']
ORIENTATION_UNDEFINED = 'UNDEFINED'
ORIENTATION_000 = 'Normal'
ORIENTATION_090 = '90 Clockwise'
ORIENTATION_180 = 'Upside-Down'
ORIENTATION_270 = '90 Counter-Clockwise'

ORIENTATION_DICT = {
    0: ORIENTATION_UNDEFINED,
    1: ORIENTATION_000,
    2: None,  # Flip Left-to-Right
    3: ORIENTATION_180,
    4: None,  # Flip Top-to-Bottom
    5: None,  # Flip Left-to-Right then Rotate 90
    6: ORIENTATION_090,
    7: None,  # Flip Left-to-Right then Rotate 270
    8: ORIENTATION_270,
}

ORIENTATION_DICT_INVERSE = {
    ORIENTATION_UNDEFINED: 0,
    ORIENTATION_000: 1,
    ORIENTATION_180: 3,
    ORIENTATION_090: 6,
    ORIENTATION_270: 8,
}

# Species
GREVYS_ZEBRA = 'grevy\'s zebra'

VALID_COLNAMES = (
    'gid',
    'uuid',
    'uri',
    'uri_original',
    'original_name',
    'ext',
    'width',
    'height',
    'time_posix',
    'gps_lat',
    'gps_lon',
    'orientation',
    'note',
    'original_path', 
    'location_code',
    'reviewed'
)