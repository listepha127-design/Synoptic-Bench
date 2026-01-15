import json
import glob
import os

INPUT_DIR = "/scratch/hay3fm/llava_training_text_synoptic/train" 


FILE_PATTERN = "*.json"

OUTPUT_FILE = "/scratch/hay3fm/llava_training_text_synoptic/llava_train_synoptic.json"

def combine_json_files(input_dir, pattern, output_file):
    """
    Finds all JSON files matching the pattern, loads them as lists, 
    and combines those lists into a single master list.
    """
    search_path = os.path.join(input_dir, pattern)
    file_paths = glob.glob(search_path)
    
    if not file_paths:
        print(f"Error: No JSON files found matching {search_path}")
        return

    print(f"Found {len(file_paths)} partial files. Combining...")

    master_data = []

    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:

                data = json.load(f) 
                
                if isinstance(data, list):
                    master_data.extend(data)
                else:
                    print(f"Warning: File {file_path} does not contain a list. Skipping.")
                    
        except json.JSONDecodeError as e:
            print(f"Error reading JSON file {file_path}: {e}")
            
    final_output_path = os.path.join(input_dir, output_file)
    with open(final_output_path, 'w', encoding='utf-8') as outfile:
        json.dump(master_data, outfile, indent=4) 

    print(f"Total training examples combined: {len(master_data)}")
    print(f"Final manifest saved to: {final_output_path}")


if __name__ == "__main__":
    combine_json_files(INPUT_DIR, FILE_PATTERN, OUTPUT_FILE)