import argparse
from datetime import datetime
from control.con_funcs import import_folder

def parse_date(date_str):
    """Parse a date string in yyyy/mm/dd format."""
    return datetime.strptime(date_str, "%Y/%m/%d")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Import directory of GGR images')
    parser.add_argument('dir_in', type=str, help='The directory to import')
    parser.add_argument('out_path', type=str, help='The full path to the .json file to store image data in')
    parser.add_argument('done_path', type=str, help='The full path to the .txt file to mark completion')
    parser.add_argument('--start_date', type=str, help='Start date in yyyy/mm/dd format')
    parser.add_argument('--end_date', type=str, help='End date in yyyy/mm/dd format')
    args = parser.parse_args()

    # Parse the date range if provided
    start_date = parse_date(args.start_date) if args.start_date else None
    end_date = parse_date(args.end_date) if args.end_date else None

    # Determine output directory and filename
    sep_idx = args.out_path.rfind("/")
    dir_out = args.out_path[:sep_idx]
    out_file = args.out_path[sep_idx:]

    # Filter images within the date range if dates are specified
    def import_folder_filtered(directory, start_date, end_date):
        images = import_folder(directory, dir_out, out_file)
        if not start_date and not end_date:
            return images
        filtered_images = []
        for image in images:
            image_date = datetime.strptime(image['date'], "%Y/%m/%d")
            if (not start_date or image_date >= start_date) and (not end_date or image_date <= end_date):
                filtered_images.append(image)
        return filtered_images

    # Import images to database with optional date filter
    image_table = import_folder_filtered(args.dir_in, start_date, end_date)
