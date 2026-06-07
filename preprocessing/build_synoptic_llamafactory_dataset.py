#!/usr/bin/env python3

"""Build LLaMA-Factory compatible Synoptic-Bench manifests.

The source JSON files keep the original ShareGPT-like `conversations` field and
the single-image `image` field. LLaMA-Factory expects a multi-modal dataset to
expose an `images` column whose paths are resolved relative to `media_dir`.

This utility rewrites the manifests so that each sample references
`<split>/<image>.png` under the chosen image root and emits a dataset_info.json
for LLaMA-Factory. By default, samples whose image file is missing locally are
omitted from the generated manifests and tracked in the build report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SPLIT_FILES = {
    "train": "training_synoptic_final.json",
    "val": "val_synoptic_final.json",
    "test": "test_synoptic_final.json",
}


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    tmp_path.replace(path)


def resolve_source_image(item: dict[str, Any]) -> str:
    if isinstance(item.get("images"), list) and item["images"]:
        raw = item["images"][0]
    else:
        raw = item.get("image")

    if not isinstance(raw, str) or not raw:
        raise ValueError(f"Missing image field in item {item.get('id')!r}.")

    return raw


def normalize_relative_image_path(raw_image: str, split: str) -> str:
    raw_path = Path(raw_image)

    if raw_path.is_absolute():
        return f"{split}/{raw_path.name}"

    if raw_path.parts and raw_path.parts[0] in SPLIT_FILES:
        return raw_path.as_posix()

    return f"{split}/{raw_path.name}"


def build_split_manifest(
    source_path: Path,
    split: str,
    image_root: Path,
    keep_missing: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with source_path.open("r", encoding="utf-8") as f:
        source_items = json.load(f)

    output_items: list[dict[str, Any]] = []
    missing_images: list[str] = []

    for item in source_items:
        raw_image = resolve_source_image(item)
        rel_image = normalize_relative_image_path(raw_image, split)
        abs_image = image_root / rel_image
        image_exists = abs_image.is_file()

        if not image_exists:
            missing_images.append(rel_image)
            if not keep_missing:
                continue

        output_items.append(
            {
                "id": item.get("id"),
                "conversations": item["conversations"],
                "images": [rel_image],
            }
        )

    report = {
        "source_file": str(source_path),
        "num_items_source": len(source_items),
        "num_items_written": len(output_items),
        "missing_images": missing_images[:50],
        "missing_image_count": len(missing_images),
    }
    return output_items, report


def build_dataset_info() -> dict[str, Any]:
    return {
        f"synoptic_{split}": {
            "file_name": f"synoptic_{split}.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "images": "images",
            },
        }
        for split in SPLIT_FILES
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_dir",
        default="/home/daxiniu12/Synoptic-Bench/hf_preview",
        help="Directory containing training/val/test Synoptic JSON files.",
    )
    parser.add_argument(
        "--image_root",
        default="/home/daxiniu12/Synoptic-Bench/data/images/American",
        help="Root directory for split subfolders with PNG images.",
    )
    parser.add_argument(
        "--output_dir",
        default="/home/daxiniu12/Synoptic-Bench/data/synoptic_lf",
        help="Output directory for LLaMA-Factory manifests and dataset_info.json.",
    )
    parser.add_argument(
        "--report_path",
        default="/home/daxiniu12/Synoptic-Bench/data/synoptic_lf/build_report.json",
        help="Path for the generation report.",
    )
    parser.add_argument(
        "--keep_missing",
        action="store_true",
        help="Keep samples whose image files are missing locally.",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)
    report_path = Path(args.report_path)

    report: dict[str, Any] = {
        "source_dir": str(source_dir),
        "image_root": str(image_root),
        "output_dir": str(output_dir),
        "keep_missing": args.keep_missing,
        "splits": {},
    }

    for split, filename in SPLIT_FILES.items():
        source_path = source_dir / filename
        manifest, split_report = build_split_manifest(source_path, split, image_root, args.keep_missing)
        atomic_write_json(output_dir / f"synoptic_{split}.json", manifest)
        report["splits"][split] = split_report

    atomic_write_json(output_dir / "dataset_info.json", build_dataset_info())
    atomic_write_json(report_path, report)

    missing_total = sum(split_report["missing_image_count"] for split_report in report["splits"].values())
    if missing_total:
        message = f"Missing {missing_total} referenced images. See {report_path} for details."
        print(f"Warning: {message}")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
