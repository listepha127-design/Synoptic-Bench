import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image
import json
import argparse
from tqdm import tqdm
import os
import warnings

warnings.filterwarnings("ignore")

LOCATIONS_JSON_PATH = "/home/hay3fm/Projects/NWS_AFD/preprocessing/locations.json"
CHECKPOINT_INTERVAL = 20

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
    MODEL_PATH = "/scratch/hay3fm/models/llava-13b"
    #MODEL_PATH = "/scratch/hay3fm/ablation/llava13B/merged-500"
    
    if not os.path.exists(MODEL_PATH):
        print(f"Local path {MODEL_PATH} not found. Trying HF Hub...")
        MODEL_PATH = "llava-hf/llava-1.5-13b-hf"

    print(f"LLaVA 1.5 13B INFERENCE")
    print(f"Loading Model: {MODEL_PATH}")
    print(f"Loading Images from: {args.image_dir}")
    
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    station_map = load_station_map(LOCATIONS_JSON_PATH)

    with open(args.manifest_json, 'r') as f:
        test_data = json.load(f)[:]
    test_data = test_data
    print(f"Data subset down to {len(test_data)} samples for ablation inference.")
        
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

    print(f"Total samples to process: {len(test_data)}")
    new_count = 0
    
    for i, sample in enumerate(tqdm(test_data, desc="LLaVA Inference")):
        if sample['id'] in processed_ids:
            continue
            
        image_filename = sample['image']
        image_path = os.path.join(args.image_dir, image_filename)
        
        if not os.path.exists(image_path):
            print(f"Skipping missing file: {image_path}")
            continue
            
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            print(f"Error opening image: {image_path}")
            continue

        correct_answer = sample.get('conversations', [{}, {'value': ''}])[1]['value']
        location_name = get_location_from_filename(image_filename, station_map)

        prompt_text = (
            f"USER: <image>\nAnalyze these weather charts showing mean 2 meter temperature over the 2-day period (shaded contours), 500 mb geopotential height (unshaded contours), and 850 mb wind velocity (wind barbs) for the {location_name} forecast region and generate a detailed forecast discussion of the large-scale features relevant to the area. The forecast region is shown in the yellow box."
            "\nASSISTANT:"
        )

        inputs = processor(
            text=prompt_text,
            images=image,
            return_tensors="pt"
        ).to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False, 
                temperature=0.1
            )

        generated_text = processor.decode(
            output_ids[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )

        results.append({
            "id": sample['id'],
            "location": location_name,
            "prediction": generated_text.strip(),
            "reference": correct_answer
        })
        
        new_count += 1

        if new_count % CHECKPOINT_INTERVAL == 0:
            with open(args.output_json, 'w') as f:
                json.dump(results, f, indent=2)

    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Done. Processed {new_count} new samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing the .png images")
    args = parser.parse_args()
    main(args)


    
