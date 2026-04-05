import json
import argparse
import evaluate
import nltk
from tqdm import tqdm
import numpy as np
from collections import Counter
import string
import sys
import re
from rouge_score import rouge_scorer

sys.setrecursionlimit(10000)

try:
    nltk.data.find('corpora/wordnet.zip')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('omw-1.4')
    nltk.download('punkt_tab')

def sanitize_text(text, max_len=2000):
    """
    Aggressively cleans text to prevent tokenizer segfaults.
    1. Truncates text longer than max_len.
    2. Detects 'repetition loops' (e.g., "cloudy cloudy cloudy...")
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    if len(text) > max_len:
        text = text[:max_len]

    text = re.sub(r'[\.\!\?]{4,}', ' ... ', text)

    tokens = text.split()
    if len(tokens) > 50:
        unique_ratio = len(set(tokens)) / len(tokens)
        if unique_ratio < 0.05: 
            return "" 

    return text

def get_stats(scores_list):
    if not scores_list:
        return {"mean": 0.0, "std": 0.0, "sem": 0.0}
    
    clean_scores = [s for s in scores_list if s is not None]
    if not clean_scores:
         return {"mean": 0.0, "std": 0.0, "sem": 0.0}

    mean_val = np.mean(clean_scores)
    std_val = np.std(clean_scores)
    sem_val = std_val / np.sqrt(len(clean_scores))
    
    return {
        "mean": float(mean_val),
        "std": float(std_val),
        "sem": float(sem_val)
    }

def print_stat_result(metric_name, stats):
    print(f"{metric_name:<12}: {stats['mean']:.4f} ± {stats['sem']:.4f} (SD: {stats['std']:.4f})")

def calculate_rouge_distribution(predictions, references):
    print("Computing ROUGE (Per-sample distribution)...")
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    r1_scores = []
    r2_scores = []
    rl_scores = []
    
    for pred, ref in tqdm(zip(predictions, references), total=len(predictions), desc="ROUGE"):
        safe_pred = sanitize_text(pred)
        safe_ref = sanitize_text(ref) 

        if not safe_pred:
            r1_scores.append(0.0)
            r2_scores.append(0.0)
            rl_scores.append(0.0)
            continue
            
        try:
            scores = scorer.score(safe_ref, safe_pred)
            r1_scores.append(scores['rouge1'].fmeasure)
            r2_scores.append(scores['rouge2'].fmeasure)
            rl_scores.append(scores['rougeL'].fmeasure)
        except Exception:
            r1_scores.append(0.0)
            r2_scores.append(0.0)
            rl_scores.append(0.0)
            
    return {
        "rouge1": get_stats(r1_scores),
        "rouge2": get_stats(r2_scores),
        "rougeL": get_stats(rl_scores)
    }

def calculate_meteor_distribution(predictions, references):
    print("Computing METEOR (Per-sample distribution)...")
    meteor = evaluate.load("meteor")
    meteor_scores = []
    
    for pred, ref in tqdm(zip(predictions, references), total=len(predictions), desc="METEOR"):
        safe_pred = sanitize_text(pred)
        if not safe_pred:
            meteor_scores.append(0.0)
            continue
        try:
            res = meteor.compute(predictions=[safe_pred], references=[ref])
            meteor_scores.append(res['meteor'])
        except:
            meteor_scores.append(0.0)
            
    return get_stats(meteor_scores)

def calculate_unigram_f1(predictions, references):
    f1_scores = []
    print("\nCalculating Unigram F1")
    for pred, ref in tqdm(zip(predictions, references), total=len(predictions), desc="Unigram F1"):
        def normalize(text):
            return text.lower().translate(str.maketrans('', '', string.punctuation)).split()
        
        pred_toks = normalize(sanitize_text(pred))
        ref_toks = normalize(ref)
        
        if len(pred_toks) == 0:
            f1_scores.append(0.0)
            continue
        common = Counter(pred_toks) & Counter(ref_toks)
        num_same = sum(common.values())
        if num_same == 0:
            f1_scores.append(0.0)
            continue
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(ref_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1)
    
    stats = get_stats(f1_scores)
    print_stat_result("Unigram F1", stats)
    return stats

def main(args):
    print(f"Loading predictions from {args.input_json}...")
    with open(args.input_json, 'r') as f:
        data = json.load(f)

    references = []
    predictions = []
    
    for item in data:
        if item.get('reference') and item.get('prediction'):
            references.append(item['reference'])
            predictions.append(item['prediction'])

    print(f"Evaluating {len(predictions)} samples...")
    full_report = {}

    rouge_stats = calculate_rouge_distribution(predictions, references)
    full_report["rouge"] = rouge_stats
    print_stat_result("ROUGE-1", rouge_stats['rouge1'])
    print_stat_result("ROUGE-2", rouge_stats['rouge2'])
    print_stat_result("ROUGE-L", rouge_stats['rougeL'])

    safe_preds_corpus = [sanitize_text(p) for p in predictions]

    meteor_stats = calculate_meteor_distribution(predictions, references)
    full_report["meteor"] = meteor_stats
    print_stat_result("METEOR", meteor_stats)

    full_report["unigram_f1"] = calculate_unigram_f1(predictions, references)

    print("\nComputing BERTScore")
    try:
        bertscore = evaluate.load("bertscore")
        bert_results = bertscore.compute(
            predictions=safe_preds_corpus, 
            references=references, 
            model_type="distilbert-base-uncased", 
            lang="en", 
            device="cuda", 
            batch_size=32 
        )
        
        f1_stats = get_stats(bert_results['f1'])
        p_stats = get_stats(bert_results['precision'])
        r_stats = get_stats(bert_results['recall'])
        
        print_stat_result("BERTScore F1", f1_stats)
        
        full_report["bertscore"] = {
            "precision": p_stats,
            "recall": r_stats,
            "f1": f1_stats
        }
        
        with open(args.output_json, 'w') as f:
            json.dump(full_report, f, indent=2)
            
    except Exception as e:
        print(f"BERTScore Failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, default="metrics_report.json")
    args = parser.parse_args()
    main(args)