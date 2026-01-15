import re
import os
from dateutil import parser
from datetime import datetime
import json

def clean_afd_text(text):
    """
    Cleans a single AFD text string and extracts metadata.
    Assumes the main "National Weather Service" header is already extracted/removed externally.
    (If not, the metadata will be empty, and the header will remain in the text).
    """
    metadata = {}
    
    text = text.replace('\x01', '').replace('\x03', '').replace('\x00', '')
    text = text.lstrip('\ufeff') # Remove Byte Order Mark if present

    lines = text.splitlines()
    cleaned_lines = []

    in_pil_block = False 
    found_main_header = False 

    for i, line in enumerate(lines):
        stripped_line = line.strip()

        if re.search(r'National Weather Service', line, re.IGNORECASE) and not found_main_header:

            office_match = re.search(r'National Weather Service\s*([A-Z/ ]+)', line, re.IGNORECASE)
            if office_match:
                metadata['issuing_office'] = office_match.group(1).strip()
            else:
                metadata['issuing_office'] = "" 

            if i + 1 < len(lines):

                pass 
            found_main_header = True 
            continue 


        if re.match(r'^\d{3}$', stripped_line) and not in_pil_block: 
            in_pil_block = True
            continue 
        elif in_pil_block and re.match(r'FXUS\d{2}\s+[A-Z]{4}\s+\d{6}', stripped_line): 
            continue 
        elif in_pil_block and re.match(r'^[A-Z]{6}$', stripped_line): 
            in_pil_block = False 
            continue 
        elif in_pil_block: 
             in_pil_block = False 

        if re.search(r'Area Forecast Discussion', line, re.IGNORECASE) or \
           re.search(r'\.\.\.New [A-Z]+\.\.\.', line, re.IGNORECASE):
            continue 

        if re.match(r'^\s*\.KEY MESSAGES\.\.\.$', stripped_line, re.IGNORECASE):
            if i + 1 < len(lines) and re.match(r'^\s*Updated at \d{4} (?:AM|PM) [A-Z]{3} \w{3} \d{1,2} \d{4}\s*$', lines[i+1].strip(), re.IGNORECASE):
                lines[i+1] = '' 
            continue 


        if re.search(r'^\s*(?:\.|[A-Z])[A-Z0-9\s\/]*?\.\.\.(?:.*?\([^\)]*\))?\s*$', stripped_line, re.IGNORECASE):
            continue 

        if re.search(r'Issued at \d{1,4}\s*(?:AM|PM)\s+[A-Z]{2,4}\s+\w{3,4}\s+\w{3,4}\s+\d{1,2}\s+\d{4}', stripped_line, re.IGNORECASE):
            continue 


        if stripped_line.lower() == 'none.' or \
           re.search(r'^[A-Z]{2,}\s*\.{2,}\s*None\.?$', stripped_line, re.IGNORECASE):
            continue
        
        cleaned_lines.append(line)

    text = '\n'.join(cleaned_lines)


    text_segments = text.split('&&')
    if len(text_segments) > 1:
        text = ' '.join(text_segments[:-1])
    else:
        text = text_segments[0]
    
    text = re.sub(r'\s*NNNN\s*', '', text).strip()
    text = re.sub(r'\s*\$\$\s*', '', text).strip()

    text = re.sub(r'\n+', ' ', text) 
    text = re.sub(r'\s+', ' ', text).strip() 

    return text, metadata

def read_text_file_with_with(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file: 
            content = file.read()
            return content
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading '{file_path}': {e}")
        return None

if __name__ == "__main__":
    input_directory = "/scratch/hay3fm/afd_data"  
    output_directory = "/scratch/hay3fm/afd_data_cleaned" 

    os.makedirs(output_directory, exist_ok=True)

    print(f"Starting cleaning process for files in '{input_directory}'...")

    processed_count = 0
    skipped_count = 0

    for filename in os.listdir(input_directory):
        if filename.endswith(".txt") and filename.startswith("AFD"):
            input_filepath = os.path.join(input_directory, filename)
            output_filepath = os.path.join(output_directory, filename) # Keep original filename

            print(f"  Processing: {filename}")
            
            file_contents = read_text_file_with_with(input_filepath)

            if file_contents is not None:
                cleaned_text, extracted_metadata = clean_afd_text(file_contents)
                
                try:
                    with open(output_filepath, 'w', encoding='utf-8') as outfile:
                        outfile.write(cleaned_text)
                    print(f"  Saved cleaned text to: {filename}")
                    processed_count += 1
                except Exception as e:
                    print(f"  Error saving cleaned text for {filename}: {e}")
                    skipped_count += 1
            else:
                print(f"  Skipping {filename} due to read error.")
                skipped_count += 1
        else:

            skipped_count += 1

    print(f"\nCleaning process complete.")
    print(f"Total files processed: {processed_count}")
    print(f"Total files skipped: {skipped_count}")
    print(f"Cleaned files are located in: {output_directory}")