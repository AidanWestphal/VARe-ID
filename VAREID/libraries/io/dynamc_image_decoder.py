import os
import cv2
import numpy as np

# DYNAMICALLY DECODE AND CROP IMAGES MAINTAINING A CACHE IN NVME STORAGE
class DynamicImageDecoder:
    def __init__(self, cache_dir=None):
        # Obtain or make cache directory (KEEP AT TMPDIR)
        self.cache_dir = cache_dir or os.environ.get('TMPDIR', '/tmp/ggr_cache')
        os.makedirs(self.cache_dir, exist_ok=True)

    # Gets a parent image from uuid
    def _get_parent_image(self, image_uuid: str, shared_memory_path: str) -> np.ndarray:
        local_full_path = os.path.join(self.cache_dir, f"full_{image_uuid}.jpg")
        
        # If the image exists in NVMe, read it
        if os.path.exists(local_full_path):
            return cv2.imread(local_full_path)
        
        # Else, read from share memory
        raw_img = cv2.imread(shared_memory_path)
        if raw_img is None:
            return np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Write to NVMe if needed
        cv2.imwrite(local_full_path, raw_img)
        return raw_img

    # Decode an annotation uuid by a given bbox
    def decode_crop(self, annot_uuid: str, image_uuid: str, shared_memory_path: str, bbox: list) -> np.ndarray:
        chip_path = os.path.join(self.cache_dir, f"chip_{annot_uuid}.jpg")
        
        # Load chip directly if cached
        if os.path.exists(chip_path):
            img = cv2.imread(chip_path)
            if img is not None: return img[:, :, ::-1]

        # If chip not loaded, get the parent image     
        raw_img = self._get_parent_image(image_uuid, shared_memory_path)
        
        # Crop parent image
        x, y, w, h = [int(v) for v in bbox]
        img_h, img_w = raw_img.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img_w, x + w), min(img_h, y + h)
        
        chip = raw_img[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Write new crop to cache
        cv2.imwrite(chip_path, chip)
        return chip[:, :, ::-1]

    # Decode a full image
    def decode_full_image(self, image_uuid: str, shared_memory_path: str) -> np.ndarray:
        return self._get_parent_image(image_uuid, shared_memory_path)[:, :, ::-1]
