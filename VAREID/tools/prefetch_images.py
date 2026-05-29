import json
import argparse
import pandas as pd
import os
from PIL import Image
import shutil
from tqdm import tqdm

def main(args):
    annot_path = args.annots
    save_dir = args.save_dir
    crop = args.crop
    print(f"LOADING ANNOTATION FILE FROM {annot_path}...")
    with open(annot_path, 'r') as f:
        data = json.load(f)
    
    # Load fields (images join w/ data)
    images = pd.DataFrame(data['images'])
    os.makedirs(save_dir, exist_ok=True)

    # If crop: loop over ANNOTS and crop each one
    if crop:
        print(f"CROPPING AND SAVING ALL ANNOTATIONS...")
        annots = pd.DataFrame(data['annotations'])
        annots.rename(columns={'uuid': 'annot_uuid'}, inplace=True)
        df = pd.merge(images, annots, how='inner', left_on='uuid', right_on='image_uuid')

        grouped = df.groupby('image_path')

        for img_path, group in tqdm(grouped, desc="Processing Images"):
            try:
                img = Image.open(img_path)
            except Exception as e:
                print(f"Failed to load {img_path}: {e}")
                continue
                
            for index, annot in group.iterrows():
                x1 = annot['bbox'][0]
                y1 = annot['bbox'][1]
                x2 = annot['bbox'][2] + x1
                y2 = annot['bbox'][3] + y1
                
                # Crop in RAM
                chip = img.crop([x1, y1, x2, y2])

                # Save cropped image by annot uuid
                fpath = os.path.join(save_dir, annot['annot_uuid'] + '.jpg')
                chip.save(fpath)
    else:
        print("SAVING ALL NON-CROPPED IMAGES...")
        for index, image in tqdm(images.iterrows(), total=len(images), desc="Copying Images"):
            fpath = os.path.join(save_dir, image['uuid'] + '.jpg')
            shutil.copy(image['image_path'], fpath)
    
    print("DONE!")

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Prefetch an annotation files images into temp memory.')
    parser.add_argument('annots', type=str, help='Path to the annots file.')
    parser.add_argument('save_dir', type=str, help='Path to the save directory.')
    parser.add_argument('-c', '--crop', action='store_true', help='Crop via bbox.')

    args = parser.parse_args()
    main(args)
