import torch
import os
import json
import argparse
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from safetensors.torch import load_file
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

LOCATIONS_JSON_PATH = "/home/hay3fm/Projects/NWS_AFD/preprocessing/locations.json"
BASE_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct" 
ADAPTER_PATH = "/scratch/hay3fm/qwen2-vl-weather-v1/checkpoint-26500" 
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
    print(f"--- QWEN2-VL FORCED SURGICAL INFERENCE ---")
    
    print("⏳ Loading Base Model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa" 
    )
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

    print("🩺 Performing surgery on Adapter weights...")
    try:
        adapter_file = os.path.join(ADAPTER_PATH, "adapter_model.safetensors")
        if not os.path.exists(adapter_file):
            adapter_file = os.path.join(ADAPTER_PATH, "adapter_model.bin")
            state_dict = torch.load(adapter_file, map_location="cpu")
        else:
            state_dict = load_file(adapter_file)
            
        clean_state_dict = {k: v for k, v in state_dict.items() if "visual" not in k and "merger" not in k}
        
        ignored_keys = len(state_dict) - len(clean_state_dict)
        print(f"   (Discarded {ignored_keys} corrupt vision keys to prevent crash)")

        model = PeftModel.from_pretrained(
            model, 
            ADAPTER_PATH, 
            is_trainable=False,
            ignore_mismatched_sizes=True
        )
        model.load_state_dict(clean_state_dict, strict=False)
        print("Surgical load complete.")
        
    except Exception as e:
        print(f"Critical Load Error: {e}")
        return

    station_map = load_station_map(LOCATIONS_JSON_PATH)
    with open(args.manifest_json, 'r') as f:
        test_data = json.load(f)

    processed_ids = set()
    results = []
    if os.path.exists(args.output_json):
        try:
            with open(args.output_json, 'r') as f:
                results = json.load(f)
                processed_ids = {item['id'] for item in results}
            print(f"Resuming... {len(processed_ids)} samples already completed.")
        except:
            print("Starting fresh.")

    model.eval()
    new_count = 0
    
    debug_printed = False

    for sample in tqdm(test_data, desc="Inference"):
        if sample['id'] in processed_ids:
            continue
            
        location_name = get_location_from_filename(sample['image'], station_map)
        full_image_path = os.path.join(args.image_dir, sample['image'])
        
        try:
            image = Image.open(full_image_path).convert("RGB")
        except:
            continue

        prompt_text = (
            f"Analyze these weather charts for {location_name}. "
            "Charts show mean 2 meter temperature (shaded), 500 mb geopotential height (contours), "
            "and 850 mb winds (barbs). Generate a detailed forecast discussion of the large-scale features."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        # Prepare Inputs
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # DEBUG: Print the first prompt to ensure image token is there
        if not debug_printed:
            print(f"\n--- PROMPT CHECK ---\n{text}\n--------------------\n")
            debug_printed = True

        inputs = processor(
            images=[image],
            text=[text],
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        # GENERATION: Aggressive settings to break the generic loop
        try:
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    min_new_tokens=10,      # FORCE it to write at least 100 tokens
                    do_sample=True,
                    temperature=0.1,         # Slightly higher temp for creativity
                    repetition_penalty=1.1  # Penalize repeating the generic phrase
                )
        except RuntimeError as e:
            print(f"⚠️ CUDA Error: {e}")
            torch.cuda.empty_cache()
            continue

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

        results.append({
            "id": sample['id'],
            "location": location_name,
            "prediction": response,
            "reference": sample.get('conversations', [{}, {'value': ''}])[1]['value']
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
    parser.add_argument("--image_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)