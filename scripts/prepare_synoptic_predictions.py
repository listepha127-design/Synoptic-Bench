#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


LOCATION_RE = re.compile(r"for the (.*?) forecast region", re.IGNORECASE)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_reference(sample: dict) -> str:
    for message in sample.get("conversations", []):
        if message.get("from") == "gpt":
            return message.get("value", "")
    return ""


def get_prompt(sample: dict) -> str:
    for message in sample.get("conversations", []):
        if message.get("from") == "human":
            return message.get("value", "")
    return ""


def get_prediction(row: dict) -> str:
    for key in ("prediction", "predict", "generated_text", "response"):
        value = row.get(key)
        if isinstance(value, str):
            return value.strip()
    return ""


def get_location(sample: dict) -> str:
    prompt = get_prompt(sample)
    match = LOCATION_RE.search(prompt)
    if match:
        return match.group(1).strip()

    sample_id = str(sample.get("id", ""))
    return sample_id.split("_", 1)[0] if sample_id else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_json", required=True)
    parser.add_argument("--predictions_jsonl", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--samples_md")
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()

    dataset_path = Path(args.dataset_json)
    predictions_path = Path(args.predictions_jsonl)
    output_path = Path(args.output_json)

    with dataset_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    predictions = load_jsonl(predictions_path)
    if len(predictions) != len(dataset):
        raise ValueError(
            f"Prediction count ({len(predictions)}) does not match dataset count ({len(dataset)})."
        )

    merged = []
    for sample, row in zip(dataset, predictions):
        merged.append(
            {
                "id": sample.get("id"),
                "location": get_location(sample),
                "reference": get_reference(sample),
                "prediction": get_prediction(row),
                "image": sample.get("images", [None])[0],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    if args.samples_md:
        samples_path = Path(args.samples_md)
        samples_path.parent.mkdir(parents=True, exist_ok=True)
        with samples_path.open("w", encoding="utf-8") as f:
            for item in merged[: args.num_samples]:
                f.write(f"## {item['id']}\n\n")
                f.write(f"Location: {item['location']}\n\n")
                f.write("Reference:\n\n")
                f.write(item["reference"].strip() + "\n\n")
                f.write("Prediction:\n\n")
                f.write(item["prediction"].strip() + "\n\n")


if __name__ == "__main__":
    main()
