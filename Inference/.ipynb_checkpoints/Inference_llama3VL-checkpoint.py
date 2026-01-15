import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
from PIL import Image
import json
import argparse
from tqdm import tqdm
import os
import warnings

# Filter warnings
warnings.filterwarnings("ignore")

# --- SETTINGS ---
LOCATIONS_JSON_PATH = "/home/hay3fm/Projects/NWS_AFD/preprocessing/locations.json"
CHECKPOINT_INTERVAL = 20
# ----------------

def load_station_map(json_path):
    if not os.path.exists(json_path): return {}
    with open(json_path, 'r') as f: return json.load(f)

def get_location_from_filename(filename, station_map):
    try:
        clean_name = os.path.splitext(os.path.basename(filename))[0]
        station_id = clean_name.split('_')[0].upper() 
        return station_map.get(station_id, f"the {station_id} forecast area")
    except:
        return "the forecast area"

def main(args):
    # Point to your local Scratch folder
    #MODEL_PATH = "/scratch/hay3fm/models/Llama-3.2-11B"
    MODEL_PATH = "/scratch/hay3fm/llama3.2-11b-weather-v1/final_merged_model-26500"
    
    # Fallback
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ Local path {MODEL_PATH} not found. Defaulting to HF Hub.")
        MODEL_PATH = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    print(f"--- LLAMA 3.2 (11B) INFERENCE ---")
    print(f"Loading Model: {MODEL_PATH}")
    print(f"Loading Images from: {args.image_dir}")
    
    # 1. LOAD MODEL
    model = MllamaForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    
    # 2. LOAD PROCESSOR
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    # 3. SETUP DATA
    station_map = load_station_map(LOCATIONS_JSON_PATH)
    with open(args.manifest_json, 'r') as f:
        test_data = json.load(f)
        
    # 4. RESUME LOGIC
    results = []
    processed_ids = set()
    if os.path.exists(args.output_json):
        try:
            with open(args.output_json, 'r') as f:
                results = json.load(f)
            for item in results:
                processed_ids.add(item['id'])
            print(f"Resuming... {len(processed_ids)} samples already completed.")
        except:
            print("Output file invalid. Starting fresh.")
    
    print(f"Processing {len(test_data)} samples...")
    new_count = 0
    
    # 5. INFERENCE LOOP
    for i, sample in enumerate(tqdm(test_data, desc="Llama Inference")):
        if sample['id'] in processed_ids:
            continue
            
        # FIX: Construct full path using the image directory
        image_filename = sample['image']
        image_path = os.path.join(args.image_dir, image_filename)
        
        correct_answer = sample.get('conversations', [{}, {'value': ''}])[1]['value']
        location_name = get_location_from_filename(image_filename, station_map)
        
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            print(f"Skipping missing: {image_path}")
            continue

        # Using a prompt very similar to your training prompt for consistency
        prompt_text = (
            f"Analyze these weather charts for the {location_name} forecast region. "
            "Charts show mean 2 meter temperature (shaded), 500 mb geopotential height (contours), "
            "and 850 mb winds (barbs). Generate a detailed forecast discussion of the large-scale features."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]

        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        
        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)

        with torch.inference_mode():
            output = model.generate(
                **inputs, 
                max_new_tokens=1024,
                do_sample=True, 
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )

        final_response = processor.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        results.append({
            "id": sample['id'],
            "location": location_name,
            "prediction": final_response.strip(),
            "reference": correct_answer
        })
        
        new_count += 1

        if new_count % CHECKPOINT_INTERVAL == 0:
            with open(args.output_json, 'w') as f:
                json.dump(results, f, indent=2)

    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    # FIX: Added image_dir argument
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing the .png images")
    args = parser.parse_args()
    main(args)