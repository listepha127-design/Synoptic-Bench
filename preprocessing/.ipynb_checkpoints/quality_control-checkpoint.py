import json
import os
from PIL import Image

# --- CONFIGURATION ---
# Change these to point to your actual data file and image directory
DATA_FILE = "/scratch/hay3fm/llava_training_text_synoptic/training_synoptic.json" # Or train.json
IMAGE_DIR = "/scratch/hay3fm/training_images_synoptic"

def run_quality_control(data_path, image_dir):
    print(f"🚀 Starting Data Quality Control on: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_samples = len(data)
    
    # Trackers for our sanity checks
    corrupted_images = 0
    missing_images = 0
    missing_image_tokens = 0
    boilerplate_found = 0
    
    # For deduplication
    unique_texts = set()
    duplicates = 0

    print(f"📊 Total samples to process: {total_samples}")

    for idx, sample in enumerate(data):
        # 1. Extract text and images based on standard LLaMA-Factory ShareGPT format
        # Adjust the keys here if your JSON uses "instruction"/"output" instead
        messages = sample.get("messages", sample.get("conversations", []))
        images = sample.get("images", [])
        
        full_text = ""
        for msg in messages:
            content = msg.get("content", msg.get("value", ""))
            full_text += content

        # 2. Check Deduplication
        sample_signature = (tuple(images), full_text)

        # 2. Check True Multimodal Deduplication
        if sample_signature in unique_texts:
            duplicates += 1
        else:
            unique_texts.add(sample_signature)

        # 3. Check for exact <image> token alignment
        if "<image>" not in full_text and len(images) > 0:
            missing_image_tokens += 1

        # 4. Check for NWS Boilerplate (&& or $$)
        if "$$" in full_text or "&&" in full_text:
            boilerplate_found += 1

        # 5. Image File Integrity Check
        for img_path in images:
            # LLaMA-Factory sometimes uses relative paths, so we join it with your media_dir
            full_img_path = os.path.join(image_dir, img_path)
            
            if not os.path.exists(full_img_path):
                missing_images += 1
                continue
                
            try:
                # Attempt to open and verify the image headers
                with Image.open(full_img_path) as img:
                    img.verify()
            except Exception:
                corrupted_images += 1

    # --- PRINT REPORT ---
    print("\n" + "="*40)
    print("📋 DATA QUALITY CONTROL REPORT")
    print("="*40)
    print(f"Total Samples Scanned:      {total_samples}")
    print(f"Exact Duplicates Found:     {duplicates}")
    print(f"Samples w/ NWS Boilerplate: {boilerplate_found} (Contains $$ or &&)")
    print(f"Samples missing <image>:    {missing_image_tokens}")
    print("-" * 40)
    print("🖼️  IMAGE HEALTH:")
    print(f"Missing Image Files:        {missing_images}")
    print(f"Corrupted Image Files:      {corrupted_images}")
    print("="*40)
    
    if corrupted_images > 0 or missing_images > 0 or duplicates > 0:
        print("⚠️  ACTION REQUIRED: You have dirty data that could impact training.")
    else:
        print("✅ DATASET PASSED SANITY CHECKS!")

if __name__ == "__main__":
    run_quality_control(DATA_FILE, IMAGE_DIR)