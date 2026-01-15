import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
import argparse
from tqdm import tqdm
import os
import warnings

warnings.filterwarnings("ignore")

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
    #MODEL_PATH = "/scratch/hay3fm/models/Qwen2-VL-7B"
    MODEL_PATH = "/scratch/hay3fm/qwen2-vl-weather-v1/final_merged_model-27000"
    
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ Local path {MODEL_PATH} not found. Defaulting to HF Hub.")
        MODEL_PATH = "Qwen/Qwen2-VL-7B-Instruct"

    print(f"--- QWEN2-VL INFERENCE ---")
    print(f"Loading Model: {MODEL_PATH}")
    print(f"Loading Images from: {args.image_dir}")
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    station_map = load_station_map(LOCATIONS_JSON_PATH)

    with open(args.manifest_json, 'r') as f:
        test_data = json.load(f)
        
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
    
    for i, sample in enumerate(tqdm(test_data, desc="Qwen Inference")):
        if sample['id'] in processed_ids:
            continue
            
        image_filename = sample['image']
        full_image_path = os.path.join(args.image_dir, image_filename)
        abs_image_path = os.path.abspath(full_image_path)
        
        if not os.path.exists(abs_image_path):
            print(f"Skipping missing file: {abs_image_path}")
            continue

        correct_answer = sample.get('conversations', [{}, {'value': ''}])[1]['value']
        location_name = get_location_from_filename(image_filename, station_map)

        prompt_text = (
            f"Analyze these weather charts for {location_name}. "
            "Charts show mean 2 meter temperature (shaded), 500 mb geopotential height (contours), "
            "and 850 mb winds (barbs). Generate a detailed forecast discussion of the large-scale features."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": abs_image_path, 
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        final_response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        results.append({
            "id": sample['id'],
            "location": location_name,
            "prediction": final_response,
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