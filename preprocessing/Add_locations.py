import json
import os
from tqdm import tqdm


# 1. Location Map
LOCATION_MAP_PATH = 'locations.json'

# 2. Input Files (The merged files you created)
INPUT_TRAIN = '/scratch/hay3fm/llava_training_text_synoptic/llava_train_synoptic_merged_all.json'
INPUT_VAL   = '/scratch/hay3fm/llava_training_text_synoptic/llava_val_synoptic_merged_all.json'
INPUT_TEST  = '/scratch/hay3fm/llava_training_text_synoptic/llava_test_synoptic_merged_all.json'

# 3. Output Files (Where the clean versions will go)
OUTPUT_TRAIN = '/scratch/hay3fm/llava_training_text_synoptic/training_synoptic.json'
OUTPUT_VAL   = '/scratch/hay3fm/llava_training_text_synoptic/val_synoptic.json'
OUTPUT_TEST  = '/scratch/hay3fm/llava_training_text_synoptic/test_synoptic.json'

# ------------------------------------------------

def load_locations(path):
    print(f"Loading location map from: {path}")
    with open(path, 'r') as f:
        return json.load(f)

def process_dataset(input_path, output_path, location_map, dataset_name="Dataset"):
    if not os.path.exists(input_path):
        print(f"Skipping {dataset_name}: File not found at {input_path}")
        return

    print(f"\n--- Processing {dataset_name} ---")
    print(f"Reading from: {input_path}")
    
    with open(input_path, 'r') as f:
        data = json.load(f)

    cleaned_data = []
    skipped_count = 0

    for sample in tqdm(data, desc=f"Cleaning {dataset_name}"):
        
        # 1. FILTER: Check text length
        # 'conversations' is a list. Index 1 is the GPT response.
        gpt_response = sample['conversations'][1]['value']
        
        if len(gpt_response) < 50:
            skipped_count += 1
            continue

        # 2. EXTRACT STATION ID
        # Format is now: STATION_Date_Sample... (e.g., UNR_nov09_2017...)
        # We need the first element (index 0)
        station_id = sample['id'].split('_')[0]

        # 3. ENRICH PROMPT
        location_name = location_map.get(station_id, "an unspecified forecast region")

        new_prompt = (
            f"<image>\nAnalyze these weather charts showing mean 2 meter temperature over the 2-day period (shaded contours), "
            f"500 mb geopotential height (unshaded contours), and 850 mb wind velocity (wind barbs) for the {location_name} "
            f"forecast region and generate a detailed forecast discussion of the large-scale features relevant to the area. "
            f"The forecast region is shown in the yellow box."
        )

        sample['conversations'][0]['value'] = new_prompt
        cleaned_data.append(sample)

    # 4. SAVE
    print(f"Original size: {len(data)}")
    print(f"Removed (text < 50 chars): {skipped_count}")
    print(f"Final size: {len(cleaned_data)}")
    print(f"Saving to: {output_path}")

    with open(output_path, 'w') as f:
        json.dump(cleaned_data, f, indent=2)

if __name__ == "__main__":
    # Load map once
    loc_map = load_locations(LOCATION_MAP_PATH)

    # Process all three splits
    #process_dataset(INPUT_TRAIN, OUTPUT_TRAIN, loc_map, "TRAIN")
    process_dataset(INPUT_VAL,   OUTPUT_VAL,   loc_map, "VAL")
    #process_dataset(INPUT_TEST,  OUTPUT_TEST,  loc_map, "TEST")
    
    print("\nAll Done!")