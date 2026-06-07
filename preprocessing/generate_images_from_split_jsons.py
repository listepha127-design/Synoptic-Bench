import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")

from prepare_dataset_in_parallel_synoptic import determine_issue_info, generate_anomaly_plot, get_var


SPLIT_FILES = {
    "train": "training_synoptic_final.json",
    "val": "val_synoptic_final.json",
    "test": "test_synoptic_final.json",
}


def parse_id(item_id):
    parts = item_id.split("_")
    if len(parts) < 5:
        return None
    sample_key = "_".join(parts[-2:])
    if not sample_key.startswith("sample_"):
        return None
    return "_".join(parts[:-4]), "_".join(parts[-4:-2]), sample_key


def decode_h5(value):
    return value.decode("utf-8") if isinstance(value, bytes) else value


def load_targets(hdf5_dir, target_ids=None):
    target_ids = set(target_ids or [])
    targets_by_sample = defaultdict(lambda: defaultdict(list))
    split_counts = {}

    for split, filename in SPLIT_FILES.items():
        json_path = hdf5_dir / filename
        with json_path.open("r") as f:
            items = json.load(f)

        split_counts[split] = len(items)
        for item in items:
            if target_ids and item.get("id") not in target_ids:
                continue
            parsed = parse_id(item.get("id", ""))
            if parsed is None:
                continue
            station_id, date_key, sample_key = parsed
            targets_by_sample[sample_key][(station_id, date_key)].append(
                {
                    "split": split,
                    "id": item["id"],
                    "image": item.get("image") or item["id"] + ".png",
                }
            )

    return targets_by_sample, split_counts


def build_plot_data(sample, climo_means, month_idx):
    raw = {"GH500": [], "U850": [], "V850": [], "t2m": []}
    for hour in range(3, 49, 3):
        lead_key = f"f_{hour}"
        if lead_key not in sample["forecasts"]:
            raise KeyError(f"missing {lead_key}")

        group = sample["forecasts"][lead_key]
        raw["GH500"].append(get_var(group, "GH500", "z"))
        raw["U850"].append(get_var(group, "U850", "u850"))
        raw["V850"].append(get_var(group, "V850", "v850"))
        raw["t2m"].append(get_var(group, "t2m", "t2m"))

    avg_t2m = np.mean(np.stack(raw["t2m"]), axis=0)
    return {
        "avg_GH500": np.mean(np.stack(raw["GH500"]), axis=0),
        "avg_U850": np.mean(np.stack(raw["U850"]), axis=0),
        "avg_V850": np.mean(np.stack(raw["V850"]), axis=0),
        "t2m_anomaly": avg_t2m - climo_means[month_idx],
    }


def collect_hdf5_files(hdf5_dir, skip_names):
    hdf5_files = []
    for path in sorted(hdf5_dir.glob("*.hdf5")):
        if path.name in skip_names:
            continue
        hdf5_files.append(path)
    return hdf5_files


def save_report(report, report_path):
    tmp_path = report_path.with_suffix(report_path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(report, f, indent=2)
    tmp_path.replace(report_path)


def generate_images(args):
    hdf5_dir = Path(args.hdf5_dir)
    output_root = Path(args.output_root)
    report_path = Path(args.report_path)

    for split in SPLIT_FILES:
        (output_root / split).mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    targets_by_sample, split_counts = load_targets(hdf5_dir, args.target_id)

    report = {
        "input_dir": str(hdf5_dir),
        "output_root": str(output_root),
        "skipped_hdf5": sorted(args.skip_hdf5),
        "json_items": split_counts,
        "splits": {
            split: {"generated": 0, "skipped_existing": 0, "failed": 0}
            for split in SPLIT_FILES
        },
        "files": {},
        "failures": [],
    }

    with h5py.File(args.climo_file, "r") as f_clim:
        climo_means = f_clim["monthly_t2m_means"][:]

    hdf5_files = collect_hdf5_files(hdf5_dir, set(args.skip_hdf5))
    target_hdf5 = args.target_hdf5
    hdf5_files = [p for p in hdf5_files if p.name == target_hdf5]
    if not hdf5_files:
        print(f"Warning: {target_hdf5} not found in {hdf5_dir}")
    for hdf5_path in hdf5_files:
        file_report = {
            "samples_seen": 0,
            "samples_matched": 0,
            "generated": 0,
            "skipped_existing": 0,
            "failed": 0,
        }
        report["files"][hdf5_path.name] = file_report
        save_report(report, report_path)

        with h5py.File(hdf5_path, "r") as f:
            lats = f["lat_global"][:]
            lons = f["lon_global"][:]
            sample_keys = sorted(k for k in f.keys() if k.startswith("sample_"))

            for sample_key in tqdm(sample_keys, desc=hdf5_path.name):
                file_report["samples_seen"] += 1
                if args.report_every and file_report["samples_seen"] % args.report_every == 0:
                    save_report(report, report_path)

                sample_targets = targets_by_sample.get(sample_key)
                if not sample_targets:
                    continue

                sample = f[sample_key]
                _, month_idx, date_key = determine_issue_info(sample["associated_afds"])
                if month_idx is None:
                    continue

                matched_afds = []
                for afd_key in sample["associated_afds"].keys():
                    afd_data = sample["associated_afds"][afd_key]
                    if "station_id" in afd_data:
                        station_id = decode_h5(afd_data["station_id"][()])
                    else:
                        station_id = afd_key

                    targets = sample_targets.get((station_id, date_key), [])
                    for target in targets:
                        matched_afds.append((afd_key, target))

                if not matched_afds:
                    continue

                file_report["samples_matched"] += 1
                try:
                    plot_data = build_plot_data(sample, climo_means, month_idx)
                except Exception as exc:
                    for _, target in matched_afds:
                        split = target["split"]
                        report["splits"][split]["failed"] += 1
                        file_report["failed"] += 1
                        if len(report["failures"]) < args.max_failures_in_report:
                            report["failures"].append(
                                {
                                    "hdf5": hdf5_path.name,
                                    "id": target["id"],
                                    "reason": str(exc),
                                }
                            )
                    continue

                for afd_key, target in matched_afds:
                    split = target["split"]
                    image_path = output_root / split / target["image"]

                    if image_path.exists() and not args.overwrite:
                        report["splits"][split]["skipped_existing"] += 1
                        file_report["skipped_existing"] += 1
                        continue

                    try:
                        afd_data = sample["associated_afds"][afd_key]
                        station_lat = afd_data["station_lat"][()]
                        station_lon = afd_data["station_lon"][()]
                        local_extent = [
                            station_lon - args.box_buffer,
                            station_lon + args.box_buffer,
                            station_lat - args.box_buffer,
                            station_lat + args.box_buffer,
                        ]
                        generate_anomaly_plot(plot_data, lats, lons, image_path, local_extent)
                        report["splits"][split]["generated"] += 1
                        file_report["generated"] += 1
                    except Exception as exc:
                        report["splits"][split]["failed"] += 1
                        file_report["failed"] += 1
                        if len(report["failures"]) < args.max_failures_in_report:
                            report["failures"].append(
                                {
                                    "hdf5": hdf5_path.name,
                                    "id": target["id"],
                                    "reason": str(exc),
                                }
                            )

                save_report(report, report_path)

        save_report(report, report_path)

    save_report(report, report_path)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_dir", default="hf_preview")
    parser.add_argument("--climo_file", default="climatology_means.h5")
    parser.add_argument("--output_root", default="data/images/American")
    parser.add_argument("--report_path", default="data/images/American/generation_report.json")
    parser.add_argument("--target_hdf5", default="training_data_2020_jul_dec.hdf5")
    parser.add_argument("--target_id", nargs="*", default=[], help="Only generate images for these manifest ids.")
    parser.add_argument(
        "--skip_hdf5",
        nargs="*",
        default=[],
        help="HDF5 file names to skip. .part files are ignored automatically.",
    )
    parser.add_argument("--box_buffer", type=float, default=2.5)
    parser.add_argument("--report_every", type=int, default=50)
    parser.add_argument("--max_failures_in_report", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    generate_images(args)


if __name__ == "__main__":
    main()
