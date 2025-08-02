"""
metrics_plots.py

Reusable utilities to visualise experiment metrics.

Each experiment lives in its own directory containing:
  • train.json – list[{"step": int, "loss": float, ...}]
  • eval.json  – list[{"dataset": str, "step": int, "accuracy": float, ...}]

Functions
---------
plot_all(
    exp_dirs: list[str],
    out_dir:  str | Path,
    avg_datasets: list[str] | None = None,
    *,
    dpi: int = 120,
    show: bool = False,
) -> None
    ▸ Generates:
        – train_loss.png
        – accuracy_<DATASET>.png  (one per dataset found)
        – average_accuracy.png   (if avg_datasets is given)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def _load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def _collect_metrics(exp_dirs: Iterable[Path]) -> tuple[
    Mapping[str, list[dict]],
    Mapping[str, dict[str, list[dict]]]
]:
    train, evals = {}, defaultdict(dict)
    for exp in exp_dirs:
        train[exp.name] = _load_json(exp / "train.json")
        for row in _load_json(exp / "eval.json"):
            evals[row["dataset"]].setdefault(exp.name, []).append(row)
    return train, evals


def _default_colors(n: int):
    return cm.tab10(np.linspace(0, 1, n))


def _plot_train_loss(
    train: Mapping[str, list[dict]],
    out_file: Path,
    dpi: int
):
    plt.figure(dpi=dpi)
    colors = _default_colors(len(train))
    for (exp_name, rows), color in zip(train.items(), colors):
        steps  = [r["step"] for r in rows]
        losses = [r["loss"] for r in rows]
        plt.plot(steps, losses, label=exp_name, color=color)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def _plot_eval_accuracy(
    evals: Mapping[str, Mapping[str, list[dict]]],
    out_dir: Path,
    dpi: int
):
    n_exp = len(next(iter(evals.values())))
    colors = _default_colors(n_exp)
    for dataset, by_exp in evals.items():
        plt.figure(dpi=dpi)
        for (exp_name, rows), color in zip(sorted(by_exp.items()), colors):
            rows_sorted = sorted(rows, key=lambda r: r["step"])
            steps = [r["step"] for r in rows_sorted]
            acc   = [r["accuracy"] for r in rows_sorted]
            plt.plot(steps, acc, label=exp_name, color=color)
        plt.xlabel("Step")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy on {dataset}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"accuracy_{dataset}.png")
        plt.close()


def _plot_average_accuracy(
    evals: Mapping[str, Mapping[str, list[dict]]],
    datasets_for_avg: list[str],
    out_file: Path,
    dpi: int
):
    missing = [d for d in datasets_for_avg if d not in evals]
    if missing:
        raise ValueError(f"Datasets not found in evals: {missing}")

    exp_names = sorted(next(iter(evals.values())).keys())
    colors = _default_colors(len(exp_names))
    plt.figure(dpi=dpi)

    # assume all datasets share the same evaluation steps
    for exp_name, color in zip(exp_names, colors):
        # step -> list[acc]
        acc_buffer: defaultdict[int, list] = defaultdict(list)
        for ds in datasets_for_avg:
            for row in evals[ds][exp_name]:
                acc_buffer[row["step"]].append(row["accuracy"])
        steps, avg_acc = zip(
            *sorted(
                (step, sum(vals) / len(vals))
                for step, vals in acc_buffer.items()
            )
        )
        plt.plot(steps, avg_acc, label=exp_name, color=color)

    plt.xlabel("Step")
    plt.ylabel(f"Average Accuracy ({len(datasets_for_avg)} datasets)")
    pretty = ", ".join(datasets_for_avg)
    plt.title(f"Average Accuracy across {pretty}")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()


def plot_all(
    exp_dirs: list[str | Path],
    out_dir: str | Path,
    avg_datasets: list[str] | None = None,
    *,
    dpi: int = 120,
    show: bool = False,
) -> None:
    exp_dirs = [Path(p).expanduser().resolve() for p in exp_dirs]
    out_dir  = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train, evals = _collect_metrics(exp_dirs)

    _plot_train_loss(train, out_dir / "train_loss.png", dpi)
    _plot_eval_accuracy(evals, out_dir, dpi)

    if avg_datasets:
        _plot_average_accuracy(
            evals,
            datasets_for_avg=avg_datasets,
            out_file=out_dir / "average_accuracy.png",
            dpi=dpi,
        )

    if show:  # optional interactive display
        import webbrowser, pathlib
        for img in out_dir.glob("*.png"):
            webbrowser.open(pathlib.Path(img).as_uri())


if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(
        description="Generate training‑loss and accuracy plots."
    )
    parser.add_argument(
        "folders",
        nargs="+",
        help="Experiment directories (each must contain train.json & eval.json)",
    )
    parser.add_argument(
        "--out",
        default="./plots",
        help="Directory where PNGs are written (default: ./plots)",
    )
    parser.add_argument(
        "--avg",
        nargs="+",
        metavar="DATASET",
        help="Datasets to average for average_accuracy.png "
             "(e.g. --avg groundui-1k webclick screenspot)",
    )
    parser.add_argument("--show", action="store_true",
                        help="Open images after creation")
    args = parser.parse_args()

    plot_all(args.folders, args.out, args.avg, show=args.show)
