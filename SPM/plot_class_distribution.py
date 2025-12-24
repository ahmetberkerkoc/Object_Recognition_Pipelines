"""Plot class-distribution as pie chart(s) + table.

Supports two modes:
1) From an existing split directory produced by `CustomDataset.xy(save_split_dir=...)`:
   <split_dir>/{train,val,test}/{class_name}/*.jpg
2) From a raw dataset root directory (folder-of-classes):
   <dataset_root>/<class_name>/*.jpg

Outputs:
- CSV table of counts
- Pie chart PNG(s)

Example:
  python3 plot_class_distribution.py --split_dir dataset2_split --out_dir save/class_dist_dataset2

"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _pretty_dataset_name(path: str) -> str:
  
    base = os.path.basename(os.path.normpath(path)).lower()
    if base.startswith("dataset1"):
        return "Dataset 1"
    if base.startswith("dataset2"):
        return "Dataset 2"
    # fallback: title-case the folder name
    return base.replace("_", " ").title()


def _safe_slug(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )


def _is_image_file(name: str) -> bool:
    return (not name.startswith(".")) and (os.path.splitext(name)[1].lower() in VALID_IMAGE_EXTS)


def _list_class_dirs(root: str) -> List[str]:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Directory not found: {root}")
    classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")]
    classes.sort()
    return classes


def _count_images_in_dir(dir_path: str) -> int:
    if not os.path.isdir(dir_path):
        return 0
    return sum(1 for f in os.listdir(dir_path) if _is_image_file(f))


@dataclass
class SplitCounts:
    classes: List[str]
    counts: Dict[str, Dict[str, int]]  # split -> class -> count

    def as_dataframe(self) -> pd.DataFrame:
        splits = list(self.counts.keys())
        data = {"class": self.classes}
        for split in splits:
            data[split] = [self.counts[split].get(c, 0) for c in self.classes]
        df = pd.DataFrame(data)
        df["total"] = df[splits].sum(axis=1)

        # Add totals row
        totals = {"class": "__TOTAL__"}
        for split in splits:
            totals[split] = int(df[split].sum())
        totals["total"] = int(df["total"].sum())
        df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)
        return df


def compute_from_split_dir(split_dir: str, splits: Tuple[str, ...] = ("train", "val", "test")) -> SplitCounts:
    classes = _list_class_dirs(os.path.join(split_dir, splits[0]))

    counts: Dict[str, Dict[str, int]] = {}
    for split in splits:
        split_root = os.path.join(split_dir, split)
        if not os.path.isdir(split_root):
            raise FileNotFoundError(f"Missing split folder: {split_root}")

        split_classes = _list_class_dirs(split_root)
        # union keeping base ordering from train folder
        for c in split_classes:
            if c not in classes:
                classes.append(c)

        counts[split] = {}
        for c in classes:
            counts[split][c] = _count_images_in_dir(os.path.join(split_root, c))

    # keep stable ordering
    classes = [c for c in classes if not c.startswith(".")]
    return SplitCounts(classes=classes, counts=counts)


def compute_from_dataset_root(dataset_root: str) -> SplitCounts:
    classes = _list_class_dirs(dataset_root)
    counts = {"all": {c: _count_images_in_dir(os.path.join(dataset_root, c)) for c in classes}}
    return SplitCounts(classes=classes, counts=counts)


def plot_pie(
    counts: Dict[str, int],
    title: str,
    out_path: str,
    *,
    show_legend: bool = True,
    legend_fontsize: int = 6,
    legend_max_cols: int = 4,
) -> None:
    labels = list(counts.keys())
    values = [counts[k] for k in labels]

    # reduce clutter: show labels only for bigger slices; always include legend
    total = sum(values) if values else 0
    if total == 0:
        raise ValueError("No images found to plot.")

    def autopct(pct: float) -> str:
        # show percent + count for slices >= 2%
        if pct < 2.0:
            return ""
        count = int(round(pct * total / 100.0))
        return f"{pct:.1f}%\n({count})"

    # Keep the pie plot size roughly the same; reserve a slim right margin for legend.
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, _, _ = ax.pie(values, startangle=90, autopct=autopct, textprops={"fontsize": 8})
    ax.axis("equal")
    ax.set_title(title)

    if show_legend:
        # legend with class + count
        legend_labels = [f"{lbl} (n={counts[lbl]})" for lbl in labels]

        # Small, compact legend on the right.
        n_items = len(legend_labels)
        ncol = min(legend_max_cols, max(1, n_items))
        ax.legend(
            wedges,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=legend_fontsize,
            ncol=ncol,
            frameon=False,
            columnspacing=0.8,
            handletextpad=0.4,
            borderaxespad=0.0,
            labelspacing=0.25,
        )

    # Leave space for the legend on the right, but keep pie area stable.
    fig.subplots_adjust(right=0.72)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dir", type=str, default=None, help="Path like dataset2_split (contains train/val/test).")
    parser.add_argument("--dataset_root", type=str, default=None, help="Path like dataset2 (folder-of-classes).")
    parser.add_argument("--out_dir", type=str, default="save/class_distribution", help="Where to write PNG/CSV.")
    parser.add_argument("--max_classes", type=int, default=0, help="If >0, keep only top-N classes by count (others merged as 'other').")

    args = parser.parse_args()

    if (args.split_dir is None) == (args.dataset_root is None):
        raise SystemExit("Provide exactly one of --split_dir or --dataset_root")

    if args.split_dir is not None:
        sc = compute_from_split_dir(args.split_dir)
        pretty = _pretty_dataset_name(args.split_dir)
        slug = _safe_slug(pretty)
    else:
        sc = compute_from_dataset_root(args.dataset_root)
        pretty = _pretty_dataset_name(args.dataset_root)
        slug = _safe_slug(pretty)

    df = sc.as_dataframe()
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"class_distribution_{slug}.csv")
    df.to_csv(csv_path, index=False)

    # plot per split
    for split, per_class in sc.counts.items():
        per_class = dict(per_class)

        if args.max_classes and args.max_classes > 0:
            # keep top-N, merge rest
            items = sorted(per_class.items(), key=lambda kv: kv[1], reverse=True)
            top = items[: args.max_classes]
            rest = items[args.max_classes :]
            per_class = dict(top)
            if rest:
                per_class["other"] = sum(v for _, v in rest)

        out_path = os.path.join(args.out_dir, f"pie_{slug}_{split}.png")
        plot_pie(
            per_class,
            title=f"{pretty} • {split}",
            out_path=out_path,
            legend_fontsize=6,
            legend_max_cols=3,
        )

    print(f"Saved CSV: {os.path.abspath(csv_path)}")
    print(f"Saved pie charts under: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
