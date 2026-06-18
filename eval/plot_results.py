"""Plot results from recall_advantage.py (Part B) or locomo_eval.py (Part A).

Auto-detects the result type from the JSON `summary` shape:
  - Part B  -> grouped bars: recall by novelty tier and by stream position.
  - Part A  -> grouped bars: accuracy by LoCoMo category.
Both annotate overall recall/accuracy and per-system cost if present.

Usage:
    python eval/plot_results.py --input eval/results/<file>.json --out fig.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _grouped_bars(ax, groups, systems, series, title, ylabel) -> None:
    import numpy as np
    x = np.arange(len(groups))
    width = 0.8 / max(1, len(systems))
    for i, system in enumerate(systems):
        # Empty bands/tiers summarize to None; plot as 0 height.
        vals = [series[system].get(g) or 0.0 for g in groups]
        ax.bar(x + i * width, vals, width, label=system)
    ax.set_xticks(x + width * (len(systems) - 1) / 2)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()


def plot_part_b(data: dict, out_path: Path) -> None:
    # Prefer a single overflow regime over the blended summary: the most-
    # overflowing (largest distractor count) condition is the discriminating one.
    by_d = data.get("summary_by_distractors") or {}
    if by_d:
        cond_key = max(by_d, key=lambda k: int(k))
        summary = by_d[cond_key]
        cond_label = f"distractors={cond_key}"
    else:
        summary = data["summary"]
        cond_label = "overall (blended)"
    systems = sorted(summary.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _grouped_bars(
        axes[0], ["high", "low"], systems,
        {s: summary[s]["by_tier"] for s in systems},
        "Recall by novelty tier", "recall@1",
    )
    _grouped_bars(
        axes[1], ["recent", "mid", "far"], systems,
        {s: summary[s]["by_position"] for s in systems},
        "Recall by tokens-after-first-mention (vs window)", "recall@1",
    )
    overall = " | ".join(f"{s}: {(summary[s]['overall'] or 0):.0%}" for s in systems)
    cost = data.get("cost", {}).get("by_system", {})
    cost_str = " | ".join(f"{s}: ${v['usd']:.2f}" for s, v in cost.items()) or "free"
    fig.suptitle(f"Part B — recall advantage [{cond_label}]   "
                 f"(recall {overall};  cost {cost_str})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"wrote {out_path}")


def plot_part_a(data: dict, out_path: Path) -> None:
    summary = data["summary"]
    systems = sorted(summary.keys())
    cats = sorted({c for s in systems for c in summary[s]["by_category"]})
    fig, ax = plt.subplots(figsize=(10, 5))
    _grouped_bars(
        ax, cats, systems,
        {s: summary[s]["by_category"] for s in systems},
        "LoCoMo accuracy by category", "accuracy",
    )
    overall = " | ".join(f"{s}: {summary[s]['overall']:.0%}" for s in systems)
    cost = data.get("cost", {}).get("by_system", {})
    cost_str = " | ".join(f"{s}: ${v['usd']:.2f}" for s, v in cost.items()) or "n/a"
    fig.suptitle(f"Part A — general parity   (overall {overall};  cost {cost_str})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Part A / Part B eval results.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    out_path = Path(args.out) if args.out else Path(args.input).with_suffix(".png")

    summary = data.get("summary", {})
    first = next(iter(summary.values()), {})
    if "by_tier" in first:
        plot_part_b(data, out_path)
    elif "by_category" in first:
        plot_part_a(data, out_path)
    else:
        raise SystemExit("Unrecognized result JSON: no by_tier or by_category in summary.")


if __name__ == "__main__":
    main()
