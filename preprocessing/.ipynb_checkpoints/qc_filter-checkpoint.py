import json
import re

# --- CONFIGURATION ---
INPUT_FILE = "/scratch/hay3fm/llava_training_text_synoptic/val_synoptic.json"
OUTPUT_FILE = "/scratch/hay3fm/llava_training_text_synoptic/val_synoptic_final.json"

MIN_WORDS = 30
MAX_WORDS = 200 # Double check if you want 200 or 800!

# Standard NWS terms indicating a forecast beyond 48 hours
LONG_TERM_KEYWORDS = [
    r"\bday 3\b", r"\bday 4\b", r"\bday 5\b", r"\bday 6\b", r"\bday 7\b",
    r"\bextended forecast\b", r"\bextended period\b", r"\bextended portion\b",
    r"\blong term\b", r"\blong-term\b", r"\blongterm\b",
    r"\bnext week\b", r"\bnext weekend\b",
    r"\bbeyond 48 hours\b", r"\bbeyond 48 hrs\b"
]

def filter_dataset():
    print(f"🚀 Loading dataset from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_original = len(data)
    filtered_data = []
    
    unique_signatures = set()
    keyword_patterns = [re.compile(kw, re.IGNORECASE) for kw in LONG_TERM_KEYWORDS]
    
    # Trackers for our final report
    dropped_duplicates = 0
    dropped_too_short = 0
    dropped_too_long = 0
    dropped_temporal_leakage = 0

    print("🧹 Filtering data... This may take a moment.")
    
    for sample in data:
        # Handle LLaMA-Factory / ShareGPT formatting
        messages = sample.get("messages", sample.get("conversations", []))
        
        # We convert the list of images to a tuple so it can be hashed in our 'set'
        images = tuple(sample.get("images", []))
        
        # 1. Extract the text to check word count
        full_text = ""
        for msg in messages:
            content = msg.get("content", msg.get("value", ""))
            full_text += content
            
        # A simple split by whitespace to approximate word count
        word_count = len(full_text.split())
        
        # 2. Apply Length Constraints
        if word_count < MIN_WORDS:
            dropped_too_short += 1
            continue
        if word_count > MAX_WORDS:
            dropped_too_long += 1
            continue
            
        # 3. Apply True Multimodal Deduplication (Image + Text)
        signature = (images, full_text)
        if signature in unique_signatures:
            dropped_duplicates += 1
            continue
            
        # 4. Temporal Leakage Filter (>48 hours)
        is_leaky = False
        for pattern in keyword_patterns:
            if pattern.search(full_text):
                is_leaky = True
                break
        
        if is_leaky:
            dropped_temporal_leakage += 1
            continue
            
        # If it survives all checks, add it to our clean dataset
        unique_signatures.add(signature)
        filtered_data.append(sample)

    # Save the new, clean dataset
    print(f"💾 Saving filtered dataset to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2)

    # Print the final methodology report
    print("\n" + "="*55)
    print("✨ DATASET FILTERING REPORT")
    print("="*55)
    print(f"Original Samples:          {total_original}")
    print(f"Dropped (Too Short <{MIN_WORDS}):   {dropped_too_short}")
    print(f"Dropped (Too Long >{MAX_WORDS}):  {dropped_too_long}")
    print(f"Dropped (Exact Duplicate): {dropped_duplicates}")
    print(f"Dropped (>48hr Leakage):   {dropped_temporal_leakage}")
    print("-" * 55)
    print(f"✅ Final Clean Dataset Size: {len(filtered_data)}")
    print("="*55)

if __name__ == "__main__":
    filter_dataset()