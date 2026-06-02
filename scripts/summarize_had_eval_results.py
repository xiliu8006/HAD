#!/usr/bin/env python
import argparse
import csv
import json
import re
from pathlib import Path


METRICS = ("psnr", "ssim", "lpips", "ellipse_time", "num_GS")
REPO_ROOT = Path(__file__).resolve().parents[1]


def latest_val_file(stats_dir: Path):
    files = list(stats_dir.glob("val_step*.json"))
    if not files:
        return None

    def step_num(path: Path) -> int:
        match = re.search(r"val_step(\d+)\.json$", path.name)
        return int(match.group(1)) if match else -1

    return max(files, key=step_num)


def load_expected(scene_list: Path | None):
    if scene_list is None:
        return []
    return [
        line.strip()
        for line in scene_list.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def collect(root: Path, step: str, expected: list[str]):
    rows = []
    for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        stats_dir = scene_dir / "stats"
        if step == "latest":
            stats_file = latest_val_file(stats_dir)
        else:
            stats_file = stats_dir / f"val_step{int(step):04d}.json"
        if stats_file is None or not stats_file.exists():
            continue
        data = json.loads(stats_file.read_text())
        row = {"scene": scene_dir.name, "stats_file": str(stats_file)}
        for metric in METRICS:
            row[metric] = data.get(metric)
        rows.append(row)

    done = {row["scene"] for row in rows}
    missing = [scene for scene in expected if scene not in done]
    means = {}
    for metric in METRICS:
        vals = [row[metric] for row in rows if isinstance(row.get(metric), (int, float))]
        means[metric] = sum(vals) / len(vals) if vals else None
    return rows, means, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=str(REPO_ROOT / "outputs" / "dreamaware3d_lvsm_view9_fusion3"),
        help="Method output directory containing scene/stats/val_step*.json files.",
    )
    parser.add_argument(
        "--scene-list",
        default=str(REPO_ROOT / "configs" / "had_eval_scenes.txt"),
        help="Optional expected scene list.",
    )
    parser.add_argument("--step", default="latest", help="'latest' or numeric step, e.g. 19999.")
    parser.add_argument("--save-json", default="", help="Optional path to save summary JSON.")
    parser.add_argument("--save-csv", default="", help="Optional path to save per-scene CSV.")
    args = parser.parse_args()

    root = Path(args.root)
    scene_list = Path(args.scene_list) if args.scene_list else None
    expected = load_expected(scene_list) if scene_list and scene_list.exists() else []
    rows, means, missing = collect(root, args.step, expected)

    print(f"root: {root}")
    print(f"done: {len(rows)}")
    if expected:
        print(f"expected: {len(expected)}")
        print(f"missing: {len(missing)}")
        for scene in missing:
            print(f"  missing {scene}")
    print("mean:")
    for metric in METRICS:
        value = means[metric]
        if value is not None:
            print(f"  {metric}: {value:.6f}")

    print("\nper_scene:")
    for row in rows:
        print(
            f"{row['scene']} "
            f"psnr={row['psnr']:.4f} "
            f"ssim={row['ssim']:.4f} "
            f"lpips={row['lpips']:.4f}"
        )

    summary = {
        "root": str(root),
        "step": args.step,
        "done": len(rows),
        "expected": len(expected),
        "missing": missing,
        "mean": means,
        "per_scene": rows,
    }
    if args.save_json:
        path = Path(args.save_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2))
        print(f"\nsaved_json: {path}")
    if args.save_csv:
        path = Path(args.save_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=("scene", *METRICS, "stats_file"))
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved_csv: {path}")


if __name__ == "__main__":
    main()
