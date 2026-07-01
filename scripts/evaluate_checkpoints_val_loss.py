#!/usr/bin/env python3
import argparse
import json
import math
import time
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


ROOT = Path(__file__).resolve().parents[1]


def load_dataset(path: Path, max_samples: int | None) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if max_samples is not None:
        data = data[:max_samples]
    return data


def get_message(sample: dict, role: str) -> str:
    from_value = "human" if role == "user" else "gpt"
    for message in sample.get("conversations", []):
        if message.get("from") == from_value:
            return message.get("value", "")
    return ""


def build_messages(sample: dict, image_path: Path) -> tuple[list[dict], list[dict]]:
    prompt = get_message(sample, "user").replace("<image>\n", "").replace("<image>", "").strip()
    answer = get_message(sample, "assistant").strip()
    user_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    full_messages = user_messages + [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
        }
    ]
    return user_messages, full_messages


def prepare_inputs(processor, sample: dict, image_root: Path, device: torch.device) -> tuple[dict, int]:
    image_rel = sample.get("images", [None])[0]
    if not image_rel:
        raise FileNotFoundError(f"{sample.get('id')} has no image reference")
    image_path = image_root / image_rel
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")

    user_messages, full_messages = build_messages(sample, image_path)
    prompt_text = processor.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)

    image_inputs, video_inputs = process_vision_info(user_messages)
    full_inputs = processor(
        text=[full_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    prompt_inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    return {k: v.to(device) for k, v in full_inputs.items()}, prompt_len


def checkpoint_step(path: Path) -> int:
    name = path.name
    if name.startswith("checkpoint-"):
        return int(name.split("-", 1)[1])
    return 10**12


def evaluate_checkpoint(args, checkpoint: Path, data: list[dict], device: torch.device) -> dict:
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, checkpoint)
    model.to(device)
    model.eval()

    total_loss_tokens = 0.0
    total_tokens = 0
    missing = 0
    start = time.time()

    with torch.inference_mode():
        for sample in tqdm(data, desc=checkpoint.name):
            try:
                inputs, prompt_len = prepare_inputs(processor, sample, Path(args.image_root), device)
            except FileNotFoundError:
                missing += 1
                continue

            labels = inputs["input_ids"].clone()
            labels[:, :prompt_len] = -100
            pad_token_id = processor.tokenizer.pad_token_id
            if pad_token_id is not None:
                labels[labels == pad_token_id] = -100
            token_count = int((labels != -100).sum().item())
            if token_count == 0:
                continue

            outputs = model(**inputs, labels=labels)
            loss = float(outputs.loss.detach().cpu().item())
            if math.isfinite(loss):
                total_loss_tokens += loss * token_count
                total_tokens += token_count

    elapsed = time.time() - start
    del model
    del base_model
    torch.cuda.empty_cache()

    val_loss = total_loss_tokens / total_tokens if total_tokens else float("nan")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_name": checkpoint.name,
        "checkpoint_step": checkpoint_step(checkpoint),
        "num_samples_requested": len(data),
        "num_missing_images": missing,
        "num_scored_samples": len(data) - missing,
        "num_label_tokens": total_tokens,
        "val_loss": val_loss,
        "perplexity": math.exp(val_loss) if math.isfinite(val_loss) and val_loss < 20 else None,
        "elapsed_seconds": elapsed,
    }


def write_results(output_path: Path, results: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranked = sorted(results, key=lambda x: x["val_loss"])
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"best": ranked[0] if ranked else None, "results": ranked}, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="/home/daxiniu12/models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--run_dir", default=str(ROOT / "saves" / "synoptic_qwen3vl_8b_lora_fa2_img64_full"))
    parser.add_argument("--dataset_json", default=str(ROOT / "data" / "synoptic_lf" / "synoptic_val.json"))
    parser.add_argument("--image_root", default=str(ROOT / "data" / "images" / "American"))
    parser.add_argument("--output_json", default=str(ROOT / "outputs" / "eval" / "synoptic_qwen3vl_8b_checkpoint_val_loss.json"))
    parser.add_argument("--checkpoints", nargs="*", help="Specific checkpoint directories. Defaults to all checkpoint-* in run_dir.")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.run_dir)
    checkpoints = [Path(p) for p in args.checkpoints] if args.checkpoints else sorted(
        [p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
        key=checkpoint_step,
    )
    data = load_dataset(Path(args.dataset_json), args.max_samples)

    output_path = Path(args.output_json)
    results = []
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            old = json.load(f)
        results = old.get("results", [])
    done = {item["checkpoint_name"] for item in results}

    for checkpoint in checkpoints:
        if checkpoint.name in done:
            print(f"Skipping completed checkpoint: {checkpoint.name}")
            continue
        print(f"Evaluating {checkpoint} on {len(data)} val samples with device={device}")
        result = evaluate_checkpoint(args, checkpoint, data, device)
        results.append(result)
        write_results(output_path, results)
        print(json.dumps(result, indent=2))

    write_results(output_path, results)
    with output_path.open("r", encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()
