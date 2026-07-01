#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINTS = [
    ROOT / "saves" / "synoptic_qwen3vl_8b_lora_fa2_img64_full" / "checkpoint-8000",
    ROOT / "saves" / "synoptic_qwen3vl_8b_lora_fa2_img64_full" / "checkpoint-10000",
    ROOT / "saves" / "synoptic_qwen3vl_8b_lora_fa2_img64_full" / "checkpoint-12000",
]
DEFAULT_DATASET_JSON = ROOT / "data" / "synoptic_lf" / "synoptic_test.json"
DEFAULT_BASE_PREDICT_YAML = (
    ROOT / "src" / "llamafactory" / "examples" / "train_lora" / "synoptic_qwen3vl_8b_lora_predict_test.yaml"
)
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "space"
DEFAULT_PREDICTIONS_ROOT = ROOT / "outputs" / "predictions"
DEFAULT_LLAMAFACTORY_DIR = ROOT / "src" / "llamafactory"
DEFAULT_LLAMAFACTORY_PYTHON = Path("/home/daxiniu12/miniconda3/envs/envqwen/bin/python")


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def checkpoint_step(checkpoint: Path) -> int:
    name = checkpoint.name
    if name.startswith("checkpoint-"):
        return int(name.split("-", 1)[1])
    raise ValueError(f"Checkpoint directory must be named checkpoint-<step>: {checkpoint}")


def prediction_output_dir(predictions_root: Path, checkpoint: Path) -> Path:
    step = checkpoint_step(checkpoint)
    return predictions_root / f"synoptic_qwen3vl_8b_lora_fa2_img64_checkpoint-{step}_test_predict"


def update_simple_yaml(src: Path, dst: Path, updates: dict[str, str | int | bool]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    out_lines = []

    with src.open("r", encoding="utf-8") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if stripped and not stripped.startswith("#") and ":" in stripped:
                key = stripped.split(":", 1)[0].strip()
                if key in updates:
                    out_lines.append(f"{key}: {format_yaml_value(updates[key])}\n")
                    seen.add(key)
                    continue
            out_lines.append(raw_line)

    missing = [key for key in updates if key not in seen]
    if missing:
        out_lines.append("\n### generated overrides\n")
        for key in missing:
            out_lines.append(f"{key}: {format_yaml_value(updates[key])}\n")

    with dst.open("w", encoding="utf-8") as f:
        f.writelines(out_lines)


def format_yaml_value(value: str | int | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def run_command(
    cmd: list[str],
    cwd: Path,
    log_path: Path,
    env: dict[str, str] | None = None,
) -> dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(cmd)}\n")
        log.write(f"cwd: {cwd}\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT)

    return {
        "command": cmd,
        "cwd": str(cwd),
        "log": str(log_path),
        "returncode": proc.returncode,
        "elapsed_seconds": time.time() - start,
    }


def llamafactory_command(args, config_path: Path) -> tuple[list[str], dict[str, str] | None]:
    if args.llamafactory_cmd:
        return args.llamafactory_cmd + ["train", str(config_path)], None

    env = os.environ.copy()
    src_path = str(args.llamafactory_dir / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    env["PATH"] = str(args.llamafactory_python.parent) + os.pathsep + env.get("PATH", "")
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    return [str(args.llamafactory_python), "-m", "llamafactory.cli", "train", str(config_path)], env


def validate_inputs(args, checkpoints: list[Path], dataset: list[dict]) -> None:
    missing = []
    for checkpoint in checkpoints:
        if not checkpoint.is_dir():
            missing.append(str(checkpoint))
    if missing:
        raise FileNotFoundError("Missing checkpoint directories:\n" + "\n".join(missing))

    if not args.dataset_json.is_file():
        raise FileNotFoundError(f"Missing dataset JSON: {args.dataset_json}")
    if not args.base_predict_yaml.is_file():
        raise FileNotFoundError(f"Missing base predict YAML: {args.base_predict_yaml}")
    if not (ROOT / "SPACE" / "location_hierarchy.json").is_file():
        raise FileNotFoundError("Missing SPACE/location_hierarchy.json")

    test_image_dir = ROOT / "data" / "images" / "American" / "test"
    if not test_image_dir.is_dir():
        raise FileNotFoundError(f"Missing test image directory: {test_image_dir}")

    sample_missing = []
    image_root = ROOT / "data" / "images" / "American"
    for sample in dataset[: min(len(dataset), 100)]:
        image_rel = sample.get("images", [sample.get("image")])[0]
        if image_rel and not (image_root / image_rel).is_file():
            sample_missing.append(str(image_root / image_rel))
            if len(sample_missing) >= 5:
                break
    if sample_missing:
        raise FileNotFoundError("Missing image files in initial test samples:\n" + "\n".join(sample_missing))


def convert_predictions(args, pred_jsonl: Path, space_input: Path, log_path: Path) -> dict:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "prepare_synoptic_predictions.py"),
        "--dataset_json",
        str(args.dataset_json),
        "--predictions_jsonl",
        str(pred_jsonl),
        "--output_json",
        str(space_input),
    ]
    return run_command(cmd, ROOT, log_path)


def run_space(space_script: Path, space_input: Path, output_json: Path, log_path: Path) -> dict:
    cmd = [
        sys.executable,
        str(space_script),
        "--input_json",
        str(space_input),
        "--output_json",
        str(output_json),
    ]
    return run_command(cmd, ROOT, log_path)


def summarize_local(local_path: Path) -> dict:
    rows = load_json(local_path)
    scores = [float(row["score"]) for row in rows]
    coverages = [float(row["coverage_ratio"]) for row in rows]
    matches = [float(row["match_score"]) for row in rows if row.get("included_in_match_avg")]
    return {
        "num_valid_samples": len(rows),
        "mean_space": mean(scores),
        "mean_coverage": mean(coverages),
        "mean_match": mean(matches),
    }


def summarize_aggregate(aggregate_path: Path) -> dict:
    data = load_json(aggregate_path)
    return data.get("summary", {})


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def validate_space_input(space_input: Path, expected_count: int) -> dict:
    rows = load_json(space_input)
    required = {"id", "location", "reference", "prediction"}
    bad_rows = [i for i, row in enumerate(rows) if not required.issubset(row)]
    nonempty_predictions = sum(1 for row in rows if str(row.get("prediction", "")).strip())
    return {
        "num_rows": len(rows),
        "expected_rows": expected_count,
        "row_count_matches": len(rows) == expected_count,
        "num_bad_rows": len(bad_rows),
        "num_nonempty_predictions": nonempty_predictions,
    }


def write_summary(output_root: Path, records: list[dict]) -> None:
    summary_json = output_root / "qwen3vl_8b_test_space_summary.json"
    summary_md = output_root / "qwen3vl_8b_test_space_summary.md"
    write_json(summary_json, {"results": records})

    output_root.mkdir(parents=True, exist_ok=True)
    with summary_md.open("w", encoding="utf-8") as f:
        f.write("# Qwen3-VL 8B Test SPACE Summary\n\n")
        f.write("| Checkpoint | Prediction rows | Local SPACE | Local coverage | Local match | Aggregate SPACE | Aggregate coverage | Aggregate match | Status |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for record in records:
            local = record.get("space_local_summary") or {}
            agg = record.get("space_aggregate_summary") or {}
            validation = record.get("space_input_validation") or {}
            f.write(
                "| {ckpt} | {rows} | {local_space} | {local_cov} | {local_match} | "
                "{agg_space} | {agg_cov} | {agg_match} | {status} |\n".format(
                    ckpt=record["checkpoint_name"],
                    rows=validation.get("num_rows", ""),
                    local_space=fmt(local.get("mean_space")),
                    local_cov=fmt(local.get("mean_coverage")),
                    local_match=fmt(local.get("mean_match")),
                    agg_space=fmt(agg.get("mean_space")),
                    agg_cov=fmt(agg.get("mean_coverage")),
                    agg_match=fmt(agg.get("mean_match")),
                    status=record.get("status", ""),
                )
            )
        f.write("\n")
        f.write(f"Summary JSON: `{summary_json}`\n")


def fmt(value) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def process_checkpoint(args, checkpoint: Path, expected_count: int) -> dict:
    checkpoint_name = checkpoint.name
    ckpt_output_root = args.output_root / checkpoint_name
    ckpt_output_root.mkdir(parents=True, exist_ok=True)

    pred_dir = prediction_output_dir(args.predictions_root, checkpoint)
    pred_jsonl = pred_dir / "generated_predictions.jsonl"
    config_path = ckpt_output_root / f"predict_{checkpoint_name}_test.yaml"
    space_input = ckpt_output_root / "test_predictions_space_input.json"
    local_json = ckpt_output_root / "space_local.json"
    aggregate_json = ckpt_output_root / "space_aggregate.json"

    record = {
        "checkpoint": str(checkpoint),
        "checkpoint_name": checkpoint_name,
        "prediction_dir": str(pred_dir),
        "prediction_jsonl": str(pred_jsonl),
        "predict_config": str(config_path),
        "space_input": str(space_input),
        "space_local_json": str(local_json),
        "space_aggregate_json": str(aggregate_json),
        "status": "started",
    }

    updates = {
        "adapter_name_or_path": checkpoint,
        "eval_dataset": "synoptic_test",
        "output_dir": pred_dir,
        "do_predict": True,
        "predict_with_generate": True,
        "max_new_tokens": 512,
        "per_device_eval_batch_size": 1,
    }
    update_simple_yaml(args.base_predict_yaml, config_path, updates)

    if args.dry_run:
        record["status"] = "dry_run"
        return record

    if args.force_predict or not pred_jsonl.is_file():
        cmd, env = llamafactory_command(args, config_path)
        record["predict"] = run_command(cmd, args.llamafactory_dir, ckpt_output_root / "predict.log", env)
        if record["predict"]["returncode"] != 0:
            record["status"] = "predict_failed"
            return record
    else:
        record["predict"] = {"skipped": True, "reason": "generated_predictions.jsonl already exists"}

    if not pred_jsonl.is_file():
        record["status"] = "prediction_missing"
        return record

    record["prediction_rows"] = count_jsonl(pred_jsonl)
    if record["prediction_rows"] != expected_count:
        record["status"] = "prediction_count_mismatch"
        if not args.allow_count_mismatch:
            return record

    if args.force_convert or not space_input.is_file():
        record["convert"] = convert_predictions(args, pred_jsonl, space_input, ckpt_output_root / "convert.log")
        if record["convert"]["returncode"] != 0:
            record["status"] = "convert_failed"
            return record
    else:
        record["convert"] = {"skipped": True, "reason": "SPACE input already exists"}

    record["space_input_validation"] = validate_space_input(space_input, expected_count)
    if (
        not record["space_input_validation"]["row_count_matches"]
        or record["space_input_validation"]["num_bad_rows"] != 0
    ) and not args.allow_count_mismatch:
        record["status"] = "space_input_invalid"
        return record

    if args.force_space or not local_json.is_file():
        record["space_local"] = run_space(
            ROOT / "SPACE" / "SPACE_local.py",
            space_input,
            local_json,
            ckpt_output_root / "space_local.log",
        )
        if record["space_local"]["returncode"] != 0:
            record["status"] = "space_local_failed"
            return record
    else:
        record["space_local"] = {"skipped": True, "reason": "space_local.json already exists"}

    if args.force_space or not aggregate_json.is_file():
        record["space_aggregate"] = run_space(
            ROOT / "SPACE" / "SPACE_aggregate.py",
            space_input,
            aggregate_json,
            ckpt_output_root / "space_aggregate.log",
        )
        if record["space_aggregate"]["returncode"] != 0:
            record["status"] = "space_aggregate_failed"
            return record
    else:
        record["space_aggregate"] = {"skipped": True, "reason": "space_aggregate.json already exists"}

    record["space_local_summary"] = summarize_local(local_json)
    record["space_aggregate_summary"] = summarize_aggregate(aggregate_json)
    record["status"] = "complete"
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run test predictions and SPACE for Qwen3-VL LoRA checkpoints.")
    parser.add_argument("--checkpoints", nargs="*", default=[str(p) for p in DEFAULT_CHECKPOINTS])
    parser.add_argument("--dataset_json", default=str(DEFAULT_DATASET_JSON))
    parser.add_argument("--base_predict_yaml", default=str(DEFAULT_BASE_PREDICT_YAML))
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--predictions_root", default=str(DEFAULT_PREDICTIONS_ROOT))
    parser.add_argument("--llamafactory_dir", default=str(DEFAULT_LLAMAFACTORY_DIR))
    parser.add_argument(
        "--llamafactory_python",
        default=str(DEFAULT_LLAMAFACTORY_PYTHON if DEFAULT_LLAMAFACTORY_PYTHON.exists() else Path(sys.executable)),
        help="Python executable used for LLaMA-Factory fallback.",
    )
    parser.add_argument("--llamafactory_cmd", nargs="+", help="Command prefix for LLaMA-Factory, e.g. llamafactory-cli")
    parser.add_argument("--cuda_visible_devices", help="CUDA_VISIBLE_DEVICES value for LLaMA-Factory prediction, e.g. 1")
    parser.add_argument("--force_predict", action="store_true", help="Re-run prediction even if JSONL exists.")
    parser.add_argument("--force_convert", action="store_true", help="Rebuild SPACE input even if it exists.")
    parser.add_argument("--force_space", action="store_true", help="Re-run SPACE even if outputs exist.")
    parser.add_argument("--allow_count_mismatch", action="store_true", help="Continue if prediction count differs from test set.")
    parser.add_argument("--dry_run", action="store_true", help="Only create/check configs; do not run prediction or SPACE.")
    args = parser.parse_args()

    args.checkpoints = [resolve_path(path) for path in args.checkpoints]
    args.dataset_json = resolve_path(args.dataset_json)
    args.base_predict_yaml = resolve_path(args.base_predict_yaml)
    args.output_root = resolve_path(args.output_root)
    args.predictions_root = resolve_path(args.predictions_root)
    args.llamafactory_dir = resolve_path(args.llamafactory_dir)
    args.llamafactory_python = resolve_path(args.llamafactory_python)
    return args


def main() -> None:
    args = parse_args()
    dataset = load_json(args.dataset_json)
    expected_count = len(dataset)
    validate_inputs(args, args.checkpoints, dataset)

    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for checkpoint in args.checkpoints:
        print(f"Processing {checkpoint.name} on synoptic_test ({expected_count} samples)", flush=True)
        record = process_checkpoint(args, checkpoint, expected_count)
        records.append(record)
        write_summary(args.output_root, records)
        print(f"{checkpoint.name}: {record['status']}", flush=True)
        if record["status"] not in {"complete", "dry_run"}:
            break

    write_summary(args.output_root, records)
    if any(record["status"] not in {"complete", "dry_run"} for record in records):
        sys.exit(1)


if __name__ == "__main__":
    main()
