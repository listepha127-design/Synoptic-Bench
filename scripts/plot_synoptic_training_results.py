#!/usr/bin/env python3
import csv
import json
import math
from pathlib import Path
from statistics import mean

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "saves" / "synoptic_qwen3vl_8b_lora_fa2_img64_full"
OUT_DIR = ROOT / "outputs" / "metrics"


COLORS = {
    "ink": "#14213d",
    "muted": "#5f6f7b",
    "grid": "#d9e1e8",
    "axis": "#203040",
    "loss": "#d1495b",
    "loss_smooth": "#00798c",
    "lr": "#edae49",
    "grad": "#3a7d44",
    "bar": "#2e86ab",
    "bar2": "#f18f01",
    "bg": "#ffffff",
    "panel": "#f6f8fa",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def parse_elapsed(value: str) -> float | None:
    if not value:
        return None
    days = 0.0
    rest = value.strip()
    if "day" in rest:
        left, rest = rest.split(",", 1)
        days = float(left.split()[0])
        rest = rest.strip()
    parts = [float(x) for x in rest.split(":")]
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = 0.0, parts[0], parts[1]
    else:
        return None
    return days * 24.0 + h + m / 60.0 + s / 3600.0


def moving_average(values: list[float], window: int) -> list[float]:
    out = []
    total = 0.0
    buf = []
    for value in values:
        buf.append(value)
        total += value
        if len(buf) > window:
            total -= buf.pop(0)
        out.append(total / len(buf))
    return out


def load_data() -> tuple[list[dict], dict, dict]:
    rows = []
    with (RUN_DIR / "trainer_log.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            row["step"] = int(row["current_steps"])
            row["elapsed_hours"] = parse_elapsed(row.get("elapsed_time", "")) or 0.0
            rows.append(row)

    with (RUN_DIR / "trainer_state.json").open("r", encoding="utf-8") as f:
        state = json.load(f)
    grad_by_step = {
        int(item["step"]): float(item["grad_norm"])
        for item in state.get("log_history", [])
        if "step" in item and "grad_norm" in item
    }
    for row in rows:
        row["grad_norm"] = grad_by_step.get(row["step"])

    with (RUN_DIR / "train_results.json").open("r", encoding="utf-8") as f:
        results = json.load(f)

    losses = [float(r["loss"]) for r in rows]
    smooth = moving_average(losses, 25)
    for row, smoothed in zip(rows, smooth):
        row["loss_smooth_500step"] = smoothed

    return rows, state, results


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        return [lo]
    span = hi - lo
    raw = span / max(n - 1, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = min([1, 2, 5, 10], key=lambda x: abs(x * mag - raw)) * mag
    start = math.floor(lo / step) * step
    ticks = []
    value = start
    while value <= hi + step * 0.5:
        if value >= lo - step * 0.2:
            ticks.append(value)
        value += step
    return ticks[: n + 2]


def draw_card(draw, box, title, value, note="", accent="#2e86ab"):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=14, fill=COLORS["panel"], outline="#d0d7de", width=2)
    draw.rectangle([x0, y0, x0 + 9, y1], fill=accent)
    draw.text((x0 + 24, y0 + 16), title, fill=COLORS["muted"], font=font(24, True))
    draw.text((x0 + 24, y0 + 50), value, fill=COLORS["ink"], font=font(38, True))
    if note:
        draw.text((x0 + 24, y0 + 100), note, fill=COLORS["muted"], font=font(20))


def draw_line_plot(
    draw,
    box,
    xs,
    series,
    title,
    xlabel,
    ylabel,
    y_limits=None,
    x_limits=None,
    checkpoints=None,
):
    x0, y0, x1, y1 = box
    margin_l, margin_r, margin_t, margin_b = 82, 30, 52, 58
    px0, py0 = x0 + margin_l, y0 + margin_t
    px1, py1 = x1 - margin_r, y1 - margin_b
    draw.rounded_rectangle(box, radius=12, fill="#ffffff", outline="#d0d7de", width=2)
    draw.text((x0 + 24, y0 + 16), title, fill=COLORS["ink"], font=font(26, True))

    xmin = min(xs) if x_limits is None else x_limits[0]
    xmax = max(xs) if x_limits is None else x_limits[1]
    all_vals = [v for _, values, _ in series for v in values if v is not None]
    ymin = min(all_vals) if y_limits is None else y_limits[0]
    ymax = max(all_vals) if y_limits is None else y_limits[1]
    if ymin == ymax:
        ymin -= 1.0
        ymax += 1.0
    pad = (ymax - ymin) * 0.08
    if y_limits is None:
        ymin -= pad
        ymax += pad

    def tx(x):
        return px0 + (x - xmin) / (xmax - xmin) * (px1 - px0)

    def ty(y):
        return py1 - (y - ymin) / (ymax - ymin) * (py1 - py0)

    for tick in nice_ticks(ymin, ymax, 5):
        yy = ty(tick)
        draw.line([px0, yy, px1, yy], fill=COLORS["grid"], width=1)
        label = f"{tick:.2g}" if abs(tick) < 0.01 else f"{tick:.2f}".rstrip("0").rstrip(".")
        draw.text((x0 + 16, yy - 10), label, fill=COLORS["muted"], font=font(18))

    for tick in nice_ticks(xmin, xmax, 6):
        xx = tx(tick)
        draw.line([xx, py0, xx, py1], fill="#eef2f5", width=1)
        draw.text((xx - 24, py1 + 14), f"{int(tick):,}", fill=COLORS["muted"], font=font(18))

    if checkpoints:
        for ckpt in checkpoints:
            if xmin <= ckpt <= xmax:
                xx = tx(ckpt)
                draw.line([xx, py0, xx, py1], fill="#b6c2cc", width=1)
                draw.text((xx + 4, py0 + 4), f"{ckpt//1000}k" if ckpt >= 1000 else str(ckpt), fill="#7a8690", font=font(16))

    draw.line([px0, py1, px1, py1], fill=COLORS["axis"], width=2)
    draw.line([px0, py0, px0, py1], fill=COLORS["axis"], width=2)
    draw.text(((px0 + px1) / 2 - 50, y1 - 34), xlabel, fill=COLORS["axis"], font=font(20, True))
    draw.text((x0 + 16, y0 + 18), ylabel, fill=COLORS["axis"], font=font(18, True))

    legend_x = x1 - 270
    legend_y = y0 + 20
    for idx, (name, values, color) in enumerate(series):
        points = []
        last = None
        for x, y in zip(xs, values):
            if y is None:
                continue
            p = (tx(x), ty(y))
            if last is not None:
                draw.line([last, p], fill=color, width=3)
            last = p
            points.append(p)
        draw.line([legend_x, legend_y + idx * 28 + 10, legend_x + 34, legend_y + idx * 28 + 10], fill=color, width=4)
        draw.text((legend_x + 44, legend_y + idx * 28), name, fill=COLORS["axis"], font=font(18))


def draw_bar_plot(draw, box, labels, values, title, ylabel):
    x0, y0, x1, y1 = box
    margin_l, margin_r, margin_t, margin_b = 72, 34, 52, 76
    px0, py0 = x0 + margin_l, y0 + margin_t
    px1, py1 = x1 - margin_r, y1 - margin_b
    draw.rounded_rectangle(box, radius=12, fill="#ffffff", outline="#d0d7de", width=2)
    draw.text((x0 + 24, y0 + 16), title, fill=COLORS["ink"], font=font(26, True))
    ymin = min(values) - 0.08
    ymax = max(values) + 0.08

    def ty(y):
        return py1 - (y - ymin) / (ymax - ymin) * (py1 - py0)

    for tick in nice_ticks(ymin, ymax, 5):
        yy = ty(tick)
        draw.line([px0, yy, px1, yy], fill=COLORS["grid"], width=1)
        draw.text((x0 + 18, yy - 10), f"{tick:.2f}", fill=COLORS["muted"], font=font(18))

    count = len(values)
    gap = 12
    width = (px1 - px0 - gap * (count - 1)) / count
    for i, (label, value) in enumerate(zip(labels, values)):
        bx0 = px0 + i * (width + gap)
        bx1 = bx0 + width
        by = ty(value)
        color = COLORS["bar"] if i < count - 1 else COLORS["bar2"]
        draw.rounded_rectangle([bx0, by, bx1, py1], radius=6, fill=color)
        draw.text((bx0 + 5, by - 26), f"{value:.2f}", fill=COLORS["axis"], font=font(17, True))
        draw.text((bx0 + 2, py1 + 14), label, fill=COLORS["muted"], font=font(15))
    draw.line([px0, py1, px1, py1], fill=COLORS["axis"], width=2)
    draw.line([px0, py0, px0, py1], fill=COLORS["axis"], width=2)
    draw.text((x0 + 16, y0 + 18), ylabel, fill=COLORS["axis"], font=font(18, True))


def stage_stats(rows: list[dict]) -> tuple[list[str], list[float]]:
    labels = []
    values = []
    for start in range(0, 100, 10):
        lo = start
        hi = start + 10
        vals = [float(r["loss"]) for r in rows if lo <= float(r["percentage"]) < hi]
        if start == 90:
            vals = [float(r["loss"]) for r in rows if 90 <= float(r["percentage"]) <= 100]
        if vals:
            labels.append(f"{lo}-{hi}%")
            values.append(mean(vals))
    return labels, values


def write_tables(rows: list[dict], summary: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "synoptic_qwen3vl_8b_training_log_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "epoch",
                "percentage",
                "elapsed_hours",
                "loss",
                "loss_smooth_500step",
                "lr",
                "grad_norm",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "step": row["step"],
                    "epoch": row["epoch"],
                    "percentage": row["percentage"],
                    "elapsed_hours": f"{row['elapsed_hours']:.4f}",
                    "loss": row["loss"],
                    "loss_smooth_500step": f"{row['loss_smooth_500step']:.6f}",
                    "lr": row["lr"],
                    "grad_norm": "" if row.get("grad_norm") is None else f"{row['grad_norm']:.6f}",
                }
            )

    with (OUT_DIR / "synoptic_qwen3vl_8b_training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def create_dashboard(rows: list[dict], results: dict, summary: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    steps = [r["step"] for r in rows]
    losses = [float(r["loss"]) for r in rows]
    smooth = [float(r["loss_smooth_500step"]) for r in rows]
    lrs = [float(r["lr"]) for r in rows]
    grads = [r.get("grad_norm") for r in rows]
    elapsed = [float(r["elapsed_hours"]) for r in rows]
    checkpoints = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 20452]

    img = Image.new("RGB", (1800, 1420), COLORS["bg"])
    draw = ImageDraw.Draw(img)
    draw.text((60, 36), "Synoptic Qwen3-VL-8B LoRA Training Results", fill=COLORS["ink"], font=font(44, True))
    draw.text(
        (60, 92),
        "1 epoch on 327,222 training samples; structured logs sampled every 20 optimization steps.",
        fill=COLORS["muted"],
        font=font(24),
    )

    first_avg = mean(losses[:10])
    last_avg = mean(losses[-10:])
    reduction = (first_avg - last_avg) / first_avg * 100.0

    cards = [
        ("Final train loss", f"{results['train_loss']:.4f}", "LLaMA-Factory aggregate", COLORS["loss"]),
        ("Last logged loss", f"{losses[-1]:.4f}", f"500-step smooth {smooth[-1]:.4f}", COLORS["loss_smooth"]),
        ("Loss reduction", f"{reduction:.1f}%", f"first 200 vs last 200 steps", COLORS["bar"]),
        ("Runtime", "6d 21h", f"{results['train_steps_per_second']:.3f} steps/s", COLORS["lr"]),
        ("Steps", f"{int(results.get('epoch', 1) * rows[-1]['total_steps']):,}", "checkpoint-20452 saved", COLORS["grad"]),
    ]
    card_w = 326
    for i, card in enumerate(cards):
        draw_card(draw, (60 + i * (card_w + 18), 140, 60 + i * (card_w + 18) + card_w, 270), *card)

    draw_line_plot(
        draw,
        (60, 310, 1080, 740),
        steps,
        [("loss", losses, COLORS["loss"]), ("500-step MA", smooth, COLORS["loss_smooth"])],
        "Training Loss Curve",
        "optimization step",
        "loss",
        checkpoints=checkpoints,
    )
    draw_line_plot(
        draw,
        (1110, 310, 1740, 740),
        steps,
        [("learning rate", lrs, COLORS["lr"])],
        "Learning Rate Schedule",
        "optimization step",
        "lr",
        y_limits=(0, max(lrs) * 1.1),
        checkpoints=checkpoints,
    )
    labels, values = stage_stats(rows)
    draw_bar_plot(
        draw,
        (60, 780, 880, 1335),
        labels,
        values,
        "Mean Loss by Training Progress",
        "mean loss",
    )
    draw_line_plot(
        draw,
        (920, 780, 1740, 1045),
        steps,
        [("grad norm", grads, COLORS["grad"])],
        "Gradient Norm",
        "optimization step",
        "grad norm",
        checkpoints=checkpoints,
    )
    draw_line_plot(
        draw,
        (920, 1070, 1740, 1335),
        steps,
        [("elapsed hours", elapsed, COLORS["bar"])],
        "Elapsed Wall Time",
        "optimization step",
        "hours",
        y_limits=(0, max(elapsed) * 1.05),
        checkpoints=checkpoints,
    )

    draw.text(
        (60, 1366),
        f"Best logged loss: {summary['best_logged_loss']:.4f} at step {summary['best_logged_loss_step']:,}. "
        f"Final checkpoint: {RUN_DIR / 'checkpoint-20452'}",
        fill=COLORS["muted"],
        font=font(22),
    )
    img.save(OUT_DIR / "synoptic_qwen3vl_8b_training_dashboard.png")


def create_focused_figures(rows: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    steps = [r["step"] for r in rows]
    losses = [float(r["loss"]) for r in rows]
    smooth = [float(r["loss_smooth_500step"]) for r in rows]
    lrs = [float(r["lr"]) for r in rows]
    checkpoints = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 20452]

    img = Image.new("RGB", (1500, 860), COLORS["bg"])
    draw = ImageDraw.Draw(img)
    draw_line_plot(
        draw,
        (45, 45, 1455, 815),
        steps,
        [("loss", losses, COLORS["loss"]), ("500-step MA", smooth, COLORS["loss_smooth"])],
        "Training Loss with Saved Checkpoints",
        "optimization step",
        "loss",
        checkpoints=checkpoints,
    )
    img.save(OUT_DIR / "synoptic_qwen3vl_8b_loss_curve.png")

    labels, values = stage_stats(rows)
    img = Image.new("RGB", (1250, 760), COLORS["bg"])
    draw = ImageDraw.Draw(img)
    draw_bar_plot(
        draw,
        (45, 45, 1205, 715),
        labels,
        values,
        "Loss Stabilization by Training Decile",
        "mean loss",
    )
    img.save(OUT_DIR / "synoptic_qwen3vl_8b_loss_by_decile.png")

    img = Image.new("RGB", (1500, 760), COLORS["bg"])
    draw = ImageDraw.Draw(img)
    draw_line_plot(
        draw,
        (45, 45, 1455, 715),
        steps,
        [("learning rate", lrs, COLORS["lr"])],
        "Cosine Learning Rate Schedule",
        "optimization step",
        "lr",
        y_limits=(0, max(lrs) * 1.1),
        checkpoints=checkpoints,
    )
    img.save(OUT_DIR / "synoptic_qwen3vl_8b_lr_schedule.png")


def main() -> None:
    rows, state, results = load_data()
    losses = [float(r["loss"]) for r in rows]
    best_idx = min(range(len(rows)), key=lambda i: losses[i])
    first_avg = mean(losses[:10])
    last_avg = mean(losses[-10:])
    summary = {
        "run_dir": str(RUN_DIR),
        "total_steps": int(rows[-1]["total_steps"]),
        "logged_points": len(rows),
        "epoch": results["epoch"],
        "train_loss": results["train_loss"],
        "train_runtime_seconds": results["train_runtime"],
        "train_runtime_hours": results["train_runtime"] / 3600.0,
        "train_samples_per_second": results["train_samples_per_second"],
        "train_steps_per_second": results["train_steps_per_second"],
        "first_200_steps_mean_loss": first_avg,
        "last_200_steps_mean_loss": last_avg,
        "loss_reduction_percent_first_to_last_200_steps": (first_avg - last_avg) / first_avg * 100.0,
        "last_logged_loss": losses[-1],
        "last_smoothed_loss_500step": rows[-1]["loss_smooth_500step"],
        "best_logged_loss": losses[best_idx],
        "best_logged_loss_step": rows[best_idx]["step"],
        "max_learning_rate": max(float(r["lr"]) for r in rows),
        "global_step": state["global_step"],
        "final_checkpoint": str(RUN_DIR / "checkpoint-20452"),
        "has_eval_loss": any("eval_loss" in item for item in state.get("log_history", [])),
    }
    write_tables(rows, summary)
    create_dashboard(rows, results, summary)
    create_focused_figures(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
