#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "saves" / "synoptic_qwen3vl_8b_lora_fa2_img64_full"
OUT_DIR = ROOT / "outputs" / "metrics"


def parse_elapsed(value: str) -> float:
    if not value:
        return 0.0
    days = 0.0
    rest = value.strip()
    if "day" in rest:
        day_part, rest = rest.split(",", 1)
        days = float(day_part.split()[0])
        rest = rest.strip()
    parts = [float(p) for p in rest.split(":")]
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours, minutes, seconds = 0.0, parts[0], parts[1]
    else:
        return 0.0
    return days * 24 + hours + minutes / 60 + seconds / 3600


def load_training_log() -> tuple[pd.DataFrame, dict, dict]:
    rows = []
    with (RUN_DIR / "trainer_log.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not {"current_steps", "loss", "lr"}.issubset(row):
                continue
            row["step"] = int(row["current_steps"])
            row["learning_rate"] = float(row["lr"])
            row["elapsed_hours"] = parse_elapsed(row.get("elapsed_time", ""))
            rows.append(row)

    df = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    df["loss_smooth_500step"] = df["loss"].rolling(window=25, min_periods=1).mean()

    with (RUN_DIR / "trainer_state.json").open("r", encoding="utf-8") as f:
        trainer_state = json.load(f)
    grad_df = pd.DataFrame(
        [
            {"step": item["step"], "grad_norm": item["grad_norm"]}
            for item in trainer_state.get("log_history", [])
            if "step" in item and "grad_norm" in item
        ]
    )
    if not grad_df.empty:
        df = df.merge(grad_df, on="step", how="left")

    with (RUN_DIR / "train_results.json").open("r", encoding="utf-8") as f:
        train_results = json.load(f)

    return df, trainer_state, train_results


def save_summary(df: pd.DataFrame, trainer_state: dict, train_results: dict) -> dict:
    first_window = df.head(10)["loss"].mean()
    last_window = df.tail(10)["loss"].mean()
    best_row = df.loc[df["loss"].idxmin()]
    summary = {
        "run_dir": str(RUN_DIR),
        "final_checkpoint": str(RUN_DIR / "checkpoint-20452"),
        "global_step": int(trainer_state["global_step"]),
        "logged_points": int(len(df)),
        "epoch": float(train_results["epoch"]),
        "train_loss": float(train_results["train_loss"]),
        "train_runtime_seconds": float(train_results["train_runtime"]),
        "train_runtime_hours": float(train_results["train_runtime"] / 3600),
        "train_samples_per_second": float(train_results["train_samples_per_second"]),
        "train_steps_per_second": float(train_results["train_steps_per_second"]),
        "first_200_steps_mean_loss": float(first_window),
        "last_200_steps_mean_loss": float(last_window),
        "loss_reduction_percent_first_to_last_200_steps": float((first_window - last_window) / first_window * 100),
        "last_logged_loss": float(df.iloc[-1]["loss"]),
        "last_smoothed_loss_500step": float(df.iloc[-1]["loss_smooth_500step"]),
        "best_logged_loss": float(best_row["loss"]),
        "best_logged_loss_step": int(best_row["step"]),
        "max_learning_rate": float(df["learning_rate"].max()),
        "has_eval_loss": bool(any("eval_loss" in item for item in trainer_state.get("log_history", []))),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "synoptic_qwen3vl_8b_training_log_summary.csv", index=False)
    with (OUT_DIR / "synoptic_qwen3vl_8b_training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def style_axes(ax):
    ax.grid(True, color="#d9e1e8", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)


def add_checkpoints(ax):
    for step in [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 20452]:
        ax.axvline(step, color="#aab6c2", linewidth=0.7, alpha=0.55)


def plot_dashboard(df: pd.DataFrame, summary: dict) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
        }
    )

    fig = plt.figure(figsize=(16, 12), constrained_layout=False)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(4, 3, height_ratios=[0.95, 2.2, 2.2, 2.2], hspace=0.52, wspace=0.32)

    fig.suptitle("Synoptic Qwen3-VL-8B LoRA Training Results", fontsize=22, fontweight="bold", x=0.05, ha="left")
    fig.text(
        0.05,
        0.945,
        "1 epoch, 20,452 optimization steps, 327,222 training samples. Curves are from trainer_log.jsonl.",
        fontsize=12,
        color="#5f6f7b",
    )

    cards = [
        ("Final train loss", f"{summary['train_loss']:.4f}", "aggregate"),
        ("Last logged loss", f"{summary['last_logged_loss']:.4f}", f"smooth {summary['last_smoothed_loss_500step']:.4f}"),
        ("Loss reduction", f"{summary['loss_reduction_percent_first_to_last_200_steps']:.1f}%", "first vs last 200 steps"),
        ("Runtime", f"{summary['train_runtime_hours']:.1f} h", "6d 21h 04m"),
        ("Throughput", f"{summary['train_steps_per_second']:.3f} step/s", f"{summary['train_samples_per_second']:.3f} sample/s"),
        ("Best logged loss", f"{summary['best_logged_loss']:.4f}", f"step {summary['best_logged_loss_step']:,}"),
    ]
    for i, (title, value, note) in enumerate(cards):
        ax = fig.add_subplot(gs[0, i % 3])
        if i >= 3:
            ax.remove()
            continue
        ax.set_facecolor("#f6f8fa")
        for spine in ax.spines.values():
            spine.set_color("#d0d7de")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.04, 0.68, title, transform=ax.transAxes, fontsize=11, color="#5f6f7b", fontweight="bold")
        ax.text(0.04, 0.28, value, transform=ax.transAxes, fontsize=22, color="#14213d", fontweight="bold")
        ax.text(0.55, 0.32, note, transform=ax.transAxes, fontsize=10, color="#5f6f7b")

    ax1 = fig.add_subplot(gs[1, :2])
    ax1.plot(df["step"], df["loss"], color="#d1495b", linewidth=1.0, alpha=0.45, label="logged loss")
    ax1.plot(df["step"], df["loss_smooth_500step"], color="#00798c", linewidth=2.2, label="500-step moving average")
    add_checkpoints(ax1)
    ax1.set_title("Training Loss Curve")
    ax1.set_xlabel("Optimization step")
    ax1.set_ylabel("Loss")
    ax1.legend(frameon=False)
    style_axes(ax1)

    ax2 = fig.add_subplot(gs[1, 2])
    ax2.plot(df["step"], df["learning_rate"], color="#edae49", linewidth=2.0)
    add_checkpoints(ax2)
    ax2.set_title("Learning Rate Schedule")
    ax2.set_xlabel("Optimization step")
    ax2.set_ylabel("Learning rate")
    style_axes(ax2)

    ax3 = fig.add_subplot(gs[2, :])
    decile = pd.cut(df["percentage"], bins=list(range(0, 101, 10)), include_lowest=True, right=False)
    decile_mean = df.groupby(decile, observed=False)["loss"].mean()
    labels = [f"{i * 10}-{(i + 1) * 10}%" for i in range(len(decile_mean))]
    ax3.bar(labels, decile_mean.values, color="#2e86ab")
    ax3.set_title("Mean Loss by Training Progress Decile")
    ax3.set_xlabel("Training progress")
    ax3.set_ylabel("Mean loss")
    ax3.tick_params(axis="x", rotation=35)
    style_axes(ax3)

    ax4 = fig.add_subplot(gs[3, :2])
    if "grad_norm" in df:
        ax4.plot(df["step"], df["grad_norm"], color="#3a7d44", linewidth=1.4)
    add_checkpoints(ax4)
    ax4.set_title("Gradient Norm")
    ax4.set_xlabel("Optimization step")
    ax4.set_ylabel("Grad norm")
    style_axes(ax4)

    ax5 = fig.add_subplot(gs[3, 2])
    ax5.plot(df["step"], df["elapsed_hours"], color="#2e86ab", linewidth=2.0)
    ax5.set_title("Elapsed Wall Time")
    ax5.set_xlabel("Optimization step")
    ax5.set_ylabel("Hours")
    style_axes(ax5)

    fig.text(
        0.05,
        0.025,
        f"Final checkpoint: {summary['final_checkpoint']}. No eval_loss was recorded in trainer_state.json.",
        fontsize=10,
        color="#5f6f7b",
    )
    fig.savefig(OUT_DIR / "synoptic_qwen3vl_8b_training_dashboard.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "synoptic_qwen3vl_8b_training_dashboard.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_single_figures(df: pd.DataFrame) -> None:
    checkpoints = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 20452]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["step"], df["loss"], color="#d1495b", linewidth=1.0, alpha=0.45, label="logged loss")
    ax.plot(df["step"], df["loss_smooth_500step"], color="#00798c", linewidth=2.2, label="500-step moving average")
    for step in checkpoints:
        ax.axvline(step, color="#aab6c2", linewidth=0.7, alpha=0.55)
    ax.set_title("Training Loss with Saved Checkpoints")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False)
    style_axes(ax)
    fig.savefig(OUT_DIR / "synoptic_qwen3vl_8b_loss_curve.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "synoptic_qwen3vl_8b_loss_curve.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    decile = pd.cut(df["percentage"], bins=list(range(0, 101, 10)), include_lowest=True, right=False)
    decile_mean = df.groupby(decile, observed=False)["loss"].mean()
    labels = [f"{i * 10}-{(i + 1) * 10}%" for i in range(len(decile_mean))]
    ax.bar(labels, decile_mean.values, color="#2e86ab")
    ax.set_title("Loss Stabilization by Training Decile")
    ax.set_xlabel("Training progress")
    ax.set_ylabel("Mean loss")
    ax.tick_params(axis="x", rotation=35)
    style_axes(ax)
    fig.savefig(OUT_DIR / "synoptic_qwen3vl_8b_loss_by_decile.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "synoptic_qwen3vl_8b_loss_by_decile.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(df["step"], df["learning_rate"], color="#edae49", linewidth=2.0)
    for step in checkpoints:
        ax.axvline(step, color="#aab6c2", linewidth=0.7, alpha=0.55)
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Learning rate")
    style_axes(ax)
    fig.savefig(OUT_DIR / "synoptic_qwen3vl_8b_lr_schedule.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "synoptic_qwen3vl_8b_lr_schedule.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, trainer_state, train_results = load_training_log()
    summary = save_summary(df, trainer_state, train_results)
    plot_dashboard(df, summary)
    plot_single_figures(df)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
