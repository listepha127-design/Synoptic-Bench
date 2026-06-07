#!/usr/bin/env python3

"""Validate images referenced by the LLaMA-Factory Synoptic manifests."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = False


def load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {path}, got {type(payload).__name__}.")

    return payload


def iter_image_refs(split: str, manifest_path: Path, media_dir: Path) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(load_manifest(manifest_path)):
        images = sample.get("images")
        if not isinstance(images, list) or not images:
            refs.append(
                {
                    "split": split,
                    "sample_index": sample_index,
                    "sample_id": sample.get("id"),
                    "image_index": None,
                    "rel_path": None,
                    "abs_path": None,
                    "status": "bad_record",
                    "error": "missing or empty images field",
                }
            )
            continue

        for image_index, rel_path in enumerate(images):
            if not isinstance(rel_path, str) or not rel_path:
                refs.append(
                    {
                        "split": split,
                        "sample_index": sample_index,
                        "sample_id": sample.get("id"),
                        "image_index": image_index,
                        "rel_path": rel_path,
                        "abs_path": None,
                        "status": "bad_record",
                        "error": "image path is not a non-empty string",
                    }
                )
                continue

            image_path = Path(rel_path)
            abs_path = image_path if image_path.is_absolute() else media_dir / image_path
            refs.append(
                {
                    "split": split,
                    "sample_index": sample_index,
                    "sample_id": sample.get("id"),
                    "image_index": image_index,
                    "rel_path": rel_path,
                    "abs_path": str(abs_path),
                }
            )

    return refs


def validate_ref(ref: dict[str, Any]) -> dict[str, Any]:
    if ref.get("status") == "bad_record":
        return ref

    abs_path = ref["abs_path"]
    path = Path(abs_path)
    if not path.is_file():
        return {**ref, "status": "missing", "error": "file does not exist"}

    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size

        with Image.open(path) as image:
            image.verify()

    except Exception as exc:  # noqa: BLE001 - report exact image decoding failure.
        return {
            **ref,
            "status": "invalid",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    return {**ref, "status": "ok", "width": width, "height": height}


def collect_refs(args: argparse.Namespace) -> list[dict[str, Any]]:
    dataset_dir = Path(args.dataset_dir)
    media_dir = Path(args.media_dir)
    refs: list[dict[str, Any]] = []

    for split in args.splits:
        manifest_path = dataset_dir / f"synoptic_{split}.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")
        split_refs = iter_image_refs(split, manifest_path, media_dir)
        if args.limit_per_split is not None:
            split_refs = split_refs[: args.limit_per_split]
        refs.extend(split_refs)

    return refs


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    by_split: dict[str, dict[str, int]] = {}
    for item in results:
        status = item["status"]
        split = item["split"]
        by_status[status] = by_status.get(status, 0) + 1
        split_counts = by_split.setdefault(split, {})
        split_counts[status] = split_counts.get(status, 0) + 1

    return {
        "total_refs": len(results),
        "by_status": by_status,
        "by_split": by_split,
        "bad_refs": [item for item in results if item["status"] != "ok"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        default="/home/daxiniu12/Synoptic-Bench/data/synoptic_lf",
    )
    parser.add_argument(
        "--media_dir",
        default="/home/daxiniu12/Synoptic-Bench/data/images/American",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--limit_per_split",
        type=int,
        default=None,
        help="Debug option: validate only this many image refs per split.",
    )
    parser.add_argument(
        "--output",
        default="/home/daxiniu12/Synoptic-Bench/outputs/logs/synoptic_image_validation_report.json",
    )
    args = parser.parse_args()

    refs = collect_refs(args)
    results: list[dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(validate_ref, ref) for ref in refs]
        for idx, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if result["status"] != "ok":
                print(json.dumps(result, ensure_ascii=False), flush=True)
            if idx % 10000 == 0:
                print(f"checked {idx}/{len(refs)} image refs", flush=True)

    report = summarize(results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps({k: v for k, v in report.items() if k != "bad_refs"}, ensure_ascii=False, indent=2))
    print(f"bad_refs: {len(report['bad_refs'])}")
    print(f"report: {output_path}")

    if report["bad_refs"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
