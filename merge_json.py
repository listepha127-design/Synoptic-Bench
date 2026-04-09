import json
import glob
import os

JSON_DIR = "/scratch/hay3fm/llava_training_text_synoptic/"


SEARCH_PATTERN = os.path.join(JSON_DIR, "llava_val_synoptic_rev_*_part_*.json")

FINAL_OUTPUT = os.path.join(JSON_DIR, "llava_val_synoptic_merged_all.json")

all_training_samples = []

chunk_files = glob.glob(SEARCH_PATTERN)
print(f"Found {len(chunk_files)} JSON chunk files to merge.")

for file in chunk_files:
    try:
        with open(file, 'r') as f:
            chunk_data = json.load(f)
            all_training_samples.extend(chunk_data) 
    except Exception as e:
        print(f"Error reading {file}: {e}")

print(f"Total training samples merged: {len(all_training_samples)}")

print("Saving master JSON file... this might take a few seconds.")
with open(FINAL_OUTPUT, 'w') as f:
    json.dump(all_training_samples, f, indent=2)

print(f"Master training file saved to: {FINAL_OUTPUT}")