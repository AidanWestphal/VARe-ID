import argparse

from control.con_funcs import import_image_folder


if __name__ == "__main__":
    # Parse command line arguments

    parser = argparse.ArgumentParser(description="Import directory of GGR images")
    parser.add_argument("dir_in", type=str, help="The directory to import")
    parser.add_argument("out_path", type=str, help="The full path to the .json file to store image data in")
    args = parser.parse_args()
    print("Importing images")

    sep_idx = args.out_path.rfind("/")
    dir_out = args.out_path[:sep_idx]
    out_file = args.out_path[sep_idx:].replace("/","")
    # Import images to database
    image_table = import_image_folder(args.dir_in, dir_out, out_file)
