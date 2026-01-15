import torch
import os
import json
import argparse
from tqdm import tqdm
from transformers import LlavaForConditionalGeneration, AutoProcessor
from peft import PeftModel
from PIL import Image
import warnings

# --- IMPORTS FOR SURGERY ---
try:
    from safetensors.torch import load_file
except ImportError:
    # Fallback if library is missing
    def load_file(f):
        raise ImportError("To load .safetensors, please run: pip install safetensors")

# Filter warnings
warnings.filterwarnings("ignore")

LOCATIONS_JSON_PATH = "/home/hay3fm/Projects/NWS_AFD/preprocessing/locations.json"
BASE_MODEL_ID = "llava-hf/llava-1.5-13b-hf" 

# UPDATE THIS: Path to your LLaVA 1.5 checkpoint
ADAPTER_PATH = "/scratch/hay3fm/llava-1.5-13b-weather-v1/checkpoint-26000" 
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
    print(f"--- LLaVA 1.5 (13B) SURGICAL INFERENCE ---")
    
    # 1. Load Base Model
    print(f"⏳ Loading Base Model: {BASE_MODEL_ID}...")
    model = LlavaForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16, 
        device_map="auto",
        attn_implementation="sdpa"
    )
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)

    # 2. SURGICAL ADAPTER LOADING
    print("🩺 Performing surgery on Adapter weights...")
    try:
        # A. Locate the weights file
        adapter_file = os.path.join(ADAPTER_PATH, "adapter_model.safetensors")
        if not os.path.exists(adapter_file):
            print("   (safetensors not found, looking for bin...)")
            adapter_file = os.path.join(ADAPTER_PATH, "adapter_model.bin")
            # If bin, load with torch
            state_dict = torch.load(adapter_file, map_location="cpu")
        else:
            # If safetensors, load with load_file
            print(f"   (Loading from {adapter_file})")
            state_dict = load_file(adapter_file)
            
        # B. FILTER OUT THE CORRUPT PROJECTOR
        # We remove any key containing 'multi_modal_projector' or 'mm_projector'
        clean_state_dict = {
            k: v for k, v in state_dict.items() 
            if "multi_modal_projector" not in k and "mm_projector" not in k
        }
        
        discarded_count = len(state_dict) - len(clean_state_dict)
        print(f"   (Discarded {discarded_count} corrupt projector keys)")

        # C. Initialize PeftModel
        model = PeftModel.from_pretrained(
            model, 
            ADAPTER_PATH, 
            is_trainable=False,
            ignore_mismatched_sizes=True # Essential to bypass the initial check
        )
        
        # D. Manually load the clean weights
        model.load_state_dict(clean_state_dict, strict=False)
        print("✅ Surgical load complete. Using Base Vision + LoRA Language.")
        
    except Exception as e:
        print(f"❌ Critical Load Error: {e}")
        return

    # 3. Setup Data
    station_map = load_station_map(LOCATIONS_JSON_PATH)
    with open(args.manifest_json, 'r') as f:
        test_data = json.load(f)

    # Resume Logic
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

    # 4. Inference Loop
    model.eval()
    new_count = 0
    
    for sample in tqdm(test_data, desc="Inference"):
        if sample['id'] in processed_ids:
            continue
            
        location_name = get_location_from_filename(sample['image'], station_map)
        full_image_path = os.path.join(args.image_dir, sample['image'])
        
        try:
            image = Image.open(full_image_path).convert("RGB")
        except:
            print(f"Skipping missing image: {full_image_path}")
            continue

        # LLaVA 1.5 Prompt Format
        prompt_user = (
            f"Analyze these weather charts for the {location_name} forecast region. "
            "Charts show mean 2 meter temperature (shaded), 500 mb geopotential height (contours), "
            "and 850 mb winds (barbs). Generate a detailed forecast discussion of the large-scale features."
        )
        formatted_prompt = f"USER: <image>\n{prompt_user}\nASSISTANT:"

        inputs = processor(
            text=formatted_prompt,
            images=image,
            return_tensors="pt"
        ).to(model.device)

        # GENERATION
        try:
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    min_new_tokens=80,       # Force longer answers
                    do_sample=True,
                    temperature=0.7,         
                    top_p=0.9,
                    repetition_penalty=1.15  
                )
        except RuntimeError as e:
            print(f"⚠️ CUDA Error on sample {sample['id']}: {e}")
            torch.cuda.empty_cache()
            continue

        # Decode
        response = processor.decode(generated_ids[0], skip_special_tokens=True)
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()

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