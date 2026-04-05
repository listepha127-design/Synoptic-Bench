#!/home/hay3fm/llava_env/bin/python
import os
import json
import argparse
import torch
import shutil
from tqdm import tqdm
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor

LOCATIONS_JSON_PATH = "/home/hay3fm/Projects/NWS_AFD/preprocessing/locations.json"
CHECKPOINT_INTERVAL = 20

def load_station_map(json_path):
    if not os.path.exists(json_path): return {}
    with open(json_path, 'r') as f: return json.load(f)

def get_location_from_filename(filename, station_map):
    try:
        clean_name = os.path.splitext(os.path.basename(filename))[0]
        parts = clean_name.split('_')
        candidate = parts[0].upper()
        if candidate.isalpha() and (3 <= len(candidate) <= 4):
            return station_map.get(candidate, f"the {candidate} forecast area")
        return "the forecast area"
    except:
        return "the forecast area"

def atomic_save(data, filepath):
    """Saves to a temp file first, then moves it. Prevents data corruption."""
    temp_path = filepath + ".tmp"
    with open(temp_path, 'w') as f:
        json.dump(data, f, indent=2)
    shutil.move(temp_path, filepath)

def main(args):
    print(f"--- STANDARD INFERENCE (HF Format) ---")
    MODEL_PATH = "llava-hf/llava-1.5-7b-hf"
    #MODEL_PATH = "/scratch/hay3fm/ablation/llava7B/merged-500"
    print(f"Loading model from: {MODEL_PATH}")
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    station_map = load_station_map(LOCATIONS_JSON_PATH)
    
    with open(args.manifest_json, 'r') as f:
        test_data = json.load(f)[:]
        
    results = []
    processed_ids = set()
    
    if os.path.exists(args.output_json):
        try:
            with open(args.output_json, 'r') as f:
                existing_data = json.load(f)
                results = existing_data
                processed_ids = {item['id'] for item in existing_data}
            print(f"🔄 Resuming: Found {len(processed_ids)} already completed items.")
        except Exception as e:

    save_counter = 0

    for sample in tqdm(test_data, desc="Generating"):
        if sample['id'] in processed_ids: 
            continue

        image_path = os.path.join(args.image_dir, sample['image'])
        if not os.path.exists(image_path): 
            continue
            
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            print(f"Error reading image: {image_path}")
            continue

        location_name = get_location_from_filename(sample['image'], station_map)
        
        user_prompt = (
            f"Analyze these weather charts showing mean 2 meter temperature over the 2-day period (shaded contours), 500 mb geopotential height (unshaded contours), and 850 mb wind velocity (wind barbs) for the {location_name} forecast region and generate a detailed forecast discussion of the large-scale features relevant to the area. The forecast region is shown in the yellow box. "
        )

        prompt = f"USER: <image>\n{user_prompt}\nASSISTANT:"

        inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda", torch.float16)

        with torch.inference_mode():
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.1,
                top_p=0.9
            )

        output_text = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
        
        if "ASSISTANT:" in output_text:
            response = output_text.split("ASSISTANT:")[-1].strip()
        else:
            response = output_text.strip()

        results.append({
            "id": sample['id'],
            "location": location_name,
            "prediction": response,
            "reference": sample.get('conversations', [{}, {'value': ''}])[1]['value']
        })
        
        save_counter += 1
        if save_counter >= CHECKPOINT_INTERVAL:
            atomic_save(results, args.output_json)
            save_counter = 0

    atomic_save(results, args.output_json)
    print(f"Results saved to {args.output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_json", type=str, required=True, help="Path to test data JSON")
    parser.add_argument("--output_json", type=str, required=True, help="Where to save results")
    parser.add_argument("--image_dir", type=str, required=True, help="Root folder of images")
    args = parser.parse_args()
    main(args)