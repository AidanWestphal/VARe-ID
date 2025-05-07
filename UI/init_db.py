from db_scripts import init_db, add_image_pair
import argparse
import os
import random

def add_random_pairs(image_dir, db_path, num_pairs=20):
    # image_files = [
    #     os.path.join(root, f)
    #     for root, _, files in os.walk(image_dir)
    #     for f in files
    #     if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    # ]

    image_files = []

    print("Walking through directory...", image_dir)

    for root, _, files in os.walk(image_dir):
        print(len(files))
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, f))

    if len(image_files) < 2:
        raise ValueError("Need at least 2 images to form a pair.")


    for _ in range(num_pairs):
        img1, img2 = random.sample(image_files, 2)
        add_image_pair(img1, img2, db_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize database and insert image pairs.")
    parser.add_argument('--db', required=True, help="Path to SQLite database file")
    parser.add_argument('--image_dir', required=True, help="Directory containing images")
    parser.add_argument('--num_pairs', type=int, default=20, help="Number of random pairs to insert")

    args = parser.parse_args()

    init_db(args.db)
    add_random_pairs(args.image_dir, args.db, args.num_pairs)