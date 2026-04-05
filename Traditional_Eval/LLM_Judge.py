import json
import os
import time
import argparse
import math
import statistics
import random
from tqdm import tqdm
import google.generativeai as genai

JUDGE_MODEL = "gemini-2.5-flash" 

def setup_judge_model():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set!")
    
    genai.configure(api_key=api_key)
    
    return genai.GenerativeModel(
        model_name=JUDGE_MODEL,
        generation_config={"response_mime_type": "application/json"}
    )

def create_evaluation_prompt(location, reference, prediction):
    return f"""You are an expert meteorologist evaluating an AI-generated weather forecast discussion.
    
Location: {location}

GROUND TRUTH FORECAST:
{reference}

AI PREDICTION:
{prediction}

INSTRUCTIONS:
Compare the generated forecast discussion to the ground truth forecast discussion. Evaluate how accurately the generated discussion captured the synoptic-scale features. Ignore minor differences in phrasing and focus on meteorological accuracy and completeness.

Score the prediction from 0 to 1:
0: Completely incorrect or severely hallucinated.
.25: Major meteorological errors or missing critical synoptic features.
.5: Partially correct, but missing some context or containing minor errors.
.75: Mostly accurate and aligns well with the ground truth, minor omissions.
1: Excellent, highly accurate, and meteorologically sound compared to the ground truth.

Respond ONLY with a valid JSON object in this exact format:
{{"score": <float>, "reasoning": "<brief string explaining the score>"}}
"""

def main(args):
    print(f"--- LLM-AS-A-JUDGE EVALUATION ---")
    model = setup_judge_model()

    if not os.path.exists(args.input_json):
        print(f"Error: Input file {args.input_json} not found.")
        return

    with open(args.input_json, 'r') as f:
        data = json.load(f)

    data = data[:10000]
    print(f"Data subset down to {len(data)} samples for evaluation.")

    evaluated_data = []
    processed_ids = set()
    
    if os.path.exists(args.output_json):
        try:
            with open(args.output_json, 'r') as f:
                evaluated_data = json.load(f)
                processed_ids = {item['id'] for item in evaluated_data}
            print(f"Resuming... {len(processed_ids)} samples already evaluated.")
        except Exception as e:
            print(f"Starting fresh. Could not load previous evaluations: {e}")

    # 3. Evaluation Loop
    save_interval = 20
    new_count = 0

    for sample in tqdm(data, desc="Judging forecasts"):
        if sample['id'] in processed_ids:
            continue

        prompt = create_evaluation_prompt(
            location=sample.get('location', 'the forecast area'),
            reference=sample.get('reference', ''),
            prediction=sample.get('prediction', '')
        )

        try:
            time.sleep(0.5) 
            
            response = model.generate_content(prompt)
            eval_result = json.loads(response.text)
            
            sample['judge_score'] = eval_result.get('score')
            sample['judge_reasoning'] = eval_result.get('reasoning')
            
            evaluated_data.append(sample)
            new_count += 1

        except Exception as e:
            print(f"\n⚠️ API Error on sample {sample['id']}: {e}")
            if "429" in str(e) or "quota" in str(e).lower():
                print("Rate limit hit. Sleeping for 60 seconds...")
                time.sleep(60)
            continue

        if new_count > 0 and new_count % save_interval == 0:
            with open(args.output_json, 'w') as f:
                json.dump(evaluated_data, f, indent=2)

    with open(args.output_json, 'w') as f:
        json.dump(evaluated_data, f, indent=2)
        
    if evaluated_data:
        valid_scores = [item['judge_score'] for item in evaluated_data if isinstance(item.get('judge_score'), (int, float))]
        n = len(valid_scores)
        
        if n > 1:
            mean_score = statistics.mean(valid_scores)
            stdev = statistics.stdev(valid_scores)
            sem = stdev / math.sqrt(n)
            
            print(f"\nEvaluation Complete! Evaluated {n} samples.")
            print(f"Mean Score: {mean_score:.4f} / 1.0")
            print(f"Standard Error of the Mean (SEM): ±{sem:.4f}")
        elif n == 1:
            print(f"\nEvaluation Complete! Only 1 sample evaluated. Score: {valid_scores[0]}")
        else:
            print("\nNo valid scores were collected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True, help="Path to your inference results JSON")
    parser.add_argument("--output_json", type=str, required=True, help="Path to save the evaluated JSON")
    args = parser.parse_args()
    main(args)