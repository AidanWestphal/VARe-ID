import os
import random
import shutil
from pathlib import Path
import cv2


def is_qr_code_image(image_path):
    """Return True if the image contains a QR code."""
    img = cv2.imread(str(image_path))
    if img is None:
        return False

    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(img)
    return data is not None and data != ""


def copy_selected_images(selected_dict, output_root):
    """Copy selected images into output_root preserving directory structure."""
    output_root = Path(output_root)

    for i, (rel_key, items) in enumerate(selected_dict.items()):
        qr = items["qr"]
        non_qr = items["non_qr"]

        # Output directory e.g. newroot/carID/camera/day
        out_dir = output_root / rel_key
        out_dir.mkdir(parents=True, exist_ok=True)

        # Copy QR
        if qr is not None:
            shutil.copy2(qr, out_dir / qr.name)

        # Copy non-QR images
        for img_path in non_qr:
            shutil.copy2(img_path, out_dir / img_path.name)
        
        print(f"Copying items ({i}/{len(selected_dict)})")


def collect_uniform_subset(
        root_dir,
        non_qr_sample_size=5,
        image_extensions=(".jpg", ".jpeg", ".png", ".heic")
    ):
    """
    Traverse root_dir and detect QR-code images inside each day folder.
    """
    root = Path(root_dir)
    results = {}
    car_ct, cam_ct, qr_ct, non_qr_ct = 0, 0, 0, 0
    car_num = 1
    car_total = len(os.listdir(root))

    for car_dir in root.iterdir():
        if not car_dir.is_dir():
            continue
        
        cam_num = 1
        cam_total = len(os.listdir(car_dir))
        for cam_dir in car_dir.iterdir():
            if not cam_dir.is_dir() or not any(["day" in day_dir.name.lower() for day_dir in cam_dir.iterdir()]):
                continue
            
            day_num = 1
            for day_dir in cam_dir.iterdir():
                if not day_dir.is_dir() or "day" not in day_dir.name.lower():
                    continue

                qr_images = []
                non_qr_images = []
                img_num = 1
                img_total = len(os.listdir(day_dir))

                # Scan images in day folder
                for img_path in day_dir.iterdir():
                    if img_path.suffix.lower() not in image_extensions or img_path.name[0] == ".":
                        continue

                    if is_qr_code_image(img_path):
                        qr_images.append(img_path)
                    else:
                        non_qr_images.append(img_path)

                    print(f"Scanning images from car ({car_num}/{car_total}), cam ({cam_num}/{cam_total}), day ({day_num}/2), image ({img_num}/{img_total})", end="\r")
                    img_num += 1

                # Choose one QR code image if present
                if qr_images:
                    selected_qr = qr_images[0]
                    qr_ct += 1
                else:
                    selected_qr = None


                # Sample up to [non_qr_sample_size] non-QR images
                selected_non_qr = random.sample(
                    non_qr_images,
                    min(non_qr_sample_size, len(non_qr_images))
                )

                non_qr_ct += len(selected_non_qr)

                # Key like "car123/camera1/day2"
                rel_key = f"{car_dir.name}/{cam_dir.name}/{day_dir.name}"
                results[rel_key] = {
                    "qr": selected_qr,
                    "non_qr": selected_non_qr
                }

                day_num += 1
            
            cam_ct += 1
            cam_num += 1
        car_ct += 1
        car_num += 1

    print(f"Collected {non_qr_ct} images and {qr_ct} QR images from {cam_ct} cameras in {car_ct} cars")

    return results


if __name__ == "__main__":
    input_root = "C:\\Users\\jmani\\Local Documents\\GGR\\Data\\GGR2024"
    output_root = "C:\\Users\\jmani\\Local Documents\\GGR\\Data\\GGR2024_Subset"

    # Collect subset
    subset = collect_uniform_subset(input_root)

    # Copy to output root
    copy_selected_images(subset, output_root)

    print("Subset copying complete.")