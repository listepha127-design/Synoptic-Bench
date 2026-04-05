import json
import os
from PIL import Image


DATA_FILE = "/scratch/hay3fm/llava_training_text_synoptic/training_synoptic.json" # Or train.json
IMAGE_DIR = "/scratch/hay3fm/training_images_synoptic"

def run_quality_control(data_path, image_dir):
    print(f"Starting Data Quality Control on: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_samples = len(data)
    
    corrupted_images = 0
    missing_images = 0
    missing_image_tokens = 0
    boilerplate_found = 0
    
    unique_texts = set()
    duplicates = 0

    print(f"Total samples to process: {total_samples}")

    for idx, sample in enumerate(data):
        messages = sample.get("messages", sample.get("conversations", []))
        images = sample.get("images", [])
        
        full_text = ""
        for msg in messages:
            content = msg.get("content", msg.get("value", ""))
            full_text += content

        sample_signature = (tuple(images), full_text)

        if sample_signature in unique_texts:
            duplicates += 1
        else:
            unique_texts.add(sample_signature)

        if "<image>" not in full_text and len(images) > 0:
            missing_image_tokens += 1

        if "$$" in full_text or "&&" in full_text:
            boilerplate_found += 1

        for img_path in images:
            full_img_path = os.path.join(image_dir, img_path)
            
            if not os.path.exists(full_img_path):
                missing_images += 1
                continue
                
            try:
                with Image.open(full_img_path) as img:
                    img.verify()
            except Exception:
                corrupted_images += 1

    print("DATA QUALITY CONTROL REPORT")
    print(f"Total Samples Scanned:      {total_samples}")
    print(f"Exact Duplicates Found:     {duplicates}")
    print(f"Samples w/ NWS Boilerplate: {boilerplate_found} (Contains $$ or &&)")
    print(f"Samples missing <image>:    {missing_image_tokens}")
    print("IMAGE HEALTH:")
    print(f"Missing Image Files:        {missing_images}")
    print(f"Corrupted Image Files:      {corrupted_images}")
    
    if corrupted_images > 0 or missing_images > 0 or duplicates > 0:
        print("ACTION REQUIRED: You have dirty data that could impact training.")
    else:
        print("DATASET PASSED SANITY CHECKS")

if __name__ == "__main__":
    run_quality_control(DATA_FILE, IMAGE_DIR)