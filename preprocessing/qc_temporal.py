import json
import random
import re

INPUT_FILE = "/scratch/hay3fm/llava_training_text_synoptic/training_synoptic.json"
SAMPLE_SIZE = 100  

LONG_TERM_KEYWORDS = [
    r"\bday 3\b", r"\bday 4\b", r"\bday 5\b", r"\bday 6\b", r"\bday 7\b",
    r"\bextended forecast\b", r"\bextended period\b", r"\bextended portion\b",
    r"\blong term\b", r"\blong-term\b", r"\blongterm\b",
    r"\bnext week\b", r"\bnext weekend\b",
    r"\bbeyond 48 hours\b", r"\bbeyond 48 hrs\b"
]

def verify_dataset():
    print(f"Loading filtered dataset from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_samples = len(data)
    leaky_samples = 0
    clean_samples_list = []
    
    keyword_patterns = [re.compile(kw, re.IGNORECASE) for kw in LONG_TERM_KEYWORDS]

    print("🕵️ Scanning for temporal leakage...")
    
    for sample in data:
        messages = sample.get("messages", sample.get("conversations", []))
        full_text = " ".join([msg.get("content", msg.get("value", "")) for msg in messages])
        
        is_leaky = False
        for pattern in keyword_patterns:
            if pattern.search(full_text):
                is_leaky = True
                break
                
        if is_leaky:
            leaky_samples += 1
        else:
            clean_samples_list.append(full_text)

    clean_count = total_samples - leaky_samples
    success_rate = (clean_count / total_samples) * 100 if total_samples > 0 else 0

    print("\n" + "="*50)
    print("TEMPORAL VERIFICATION REPORT")
    print("="*50)
    print(f"Total Samples Scanned:      {total_samples}")
    print(f"Samples with >48hr leakage: {leaky_samples}")
    print(f"Clean Samples (<48hr):      {clean_count}")
    print("-" * 50)
    print(f"Heuristic Success Rate:    {success_rate:.2f}%")
    print("="*50)

    print(f"\nGenerating {SAMPLE_SIZE} random clean samples for manual qualitative review...")
    print("Save these to a text file and read through them to verify no sneaky >48hr terms slipped through.")
    
    # Safely get up to SAMPLE_SIZE random samples
    actual_sample_size = min(SAMPLE_SIZE, len(clean_samples_list))
    random_manual_check = random.sample(clean_samples_list, actual_sample_size)
    
    with open("qualitative_review_samples.txt", "w", encoding="utf-8") as f:
        for i, text in enumerate(random_manual_check):
            f.write(f"SAMPLE {i+1} ---\n{text}\n\n")
            
    print("Saved manual review samples to 'qualitative_review_samples.txt'.")

if __name__ == "__main__":
    verify_dataset()