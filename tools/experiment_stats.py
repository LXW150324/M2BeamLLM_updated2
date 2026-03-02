#!/usr/bin/env python3
"""
Aggregate experiment result JSON files and run paired Wilcoxon signed-rank tests.

Expected input files are train_robust.py artifacts saved in logs/, e.g.:
  final_metrics_standard_none_seed20260222_lrat100_hist_ar.json

Examples:
  python tools/experiment_stats.py --glob-a "logs/final_metrics_*_hist_ar*.json"
  python tools/experiment_stats.py --glob-a "logs/final_metrics_*baseline*.json" \
      --glob-b "logs/final_metrics_*hist_ar*.json" --metric final_metrics.top_1_acc
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


def _get_nested(d: Dict, path: str):
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(f"Path '{path}' missing at '{key}'")
        cur = cur[key]
    return cur


@dataclass
class Record:
    path: str
    seed: Optional[int]
    value: float
    payload: Dict


def load_records(glob_pattern: str, metric_path: str) -> List[Record]:
    paths = sorted(glob.glob(glob_pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched: {glob_pattern}")
    recs: List[Record] = []
    for p in paths:
        with open(p, "r") as f:
            payload = json.load(f)
        seed = None
        if isinstance(payload, dict):
            seed = payload.get("meta", {}).get("seed")
        value = float(_get_nested(payload, metric_path))
        recs.append(Record(path=p, seed=seed, value=value, payload=payload))
    return recs


def summarize(values: List[float]) -> Dict[str, float]:
    n = len(values)
    if n == 0:
        return {"n": 0}
    mean = sum(values) / n
    if n > 1:
        var = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    ci95 = 1.96 * std / math.sqrt(n) if n > 0 else 0.0
    return {"n": n, "mean": mean, "std": std, "ci95": ci95}


def _rank_abs_with_ties(abs_vals: List[float]) -> Tuple[List[float], float]:
    """
    Return average ranks (1..n) and tie correction term sum(t^3 - t).
    """
    idx_sorted = sorted(range(len(abs_vals)), key=lambda i: abs_vals[i])
    ranks = [0.0] * len(abs_vals)
    tie_correction = 0.0
    i = 0
    rank = 1
    while i < len(abs_vals):
        j = i + 1
        while j < len(abs_vals) and abs_vals[idx_sorted[j]] == abs_vals[idx_sorted[i]]:
            j += 1
        count = j - i
        avg_rank = (rank + (rank + count - 1)) / 2.0
        for k in range(i, j):
            ranks[idx_sorted[k]] = avg_rank
        if count > 1:
            tie_correction += count ** 3 - count
        rank += count
        i = j
    return ranks, tie_correction


def wilcoxon_signed_rank(x: List[float], y: List[float]) -> Dict[str, float]:
    """
    Paired two-sided Wilcoxon signed-rank test.
    Uses SciPy if available; otherwise falls back to normal approximation.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    try:
        from scipy.stats import wilcoxon as scipy_wilcoxon  # type: ignore
        stat, p = scipy_wilcoxon(x, y, zero_method="wilcox", alternative="two-sided", mode="auto")
        return {"statistic": float(stat), "pvalue": float(p), "method": "scipy"}
    except Exception:
        pass

    diffs = [a - b for a, b in zip(x, y)]
    diffs = [d for d in diffs if d != 0.0]
    n = len(diffs)
    if n == 0:
        return {"statistic": 0.0, "pvalue": 1.0, "method": "fallback_all_zero"}

    abs_vals = [abs(d) for d in diffs]
    ranks, tie_corr = _rank_abs_with_ties(abs_vals)
    w_pos = sum(r for r, d in zip(ranks, diffs) if d > 0)
    w_neg = sum(r for r, d in zip(ranks, diffs) if d < 0)
    w = min(w_pos, w_neg)
    mu = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    # Tie correction for Wilcoxon signed-rank variance
    var -= tie_corr / 48.0
    var = max(var, 1e-12)
    sigma = math.sqrt(var)
    # continuity correction
    cc = 0.5 if w != mu else 0.0
    z = (w - mu + (cc if w < mu else -cc)) / sigma
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return {
        "statistic": float(w),
        "pvalue": float(p),
        "z": float(z),
        "method": "normal_approx_fallback",
        "n_nonzero_pairs": int(n),
    }


def pair_by_seed(a: List[Record], b: List[Record]) -> Tuple[List[Tuple[Record, Record]], List[int]]:
    a_by_seed = {r.seed: r for r in a if r.seed is not None}
    b_by_seed = {r.seed: r for r in b if r.seed is not None}
    common = sorted(set(a_by_seed.keys()) & set(b_by_seed.keys()))
    return [(a_by_seed[s], b_by_seed[s]) for s in common], common


def main():
    ap = argparse.ArgumentParser(description="Aggregate metrics and run Wilcoxon signed-rank.")
    ap.add_argument("--glob-a", required=True, help="Glob for group A JSON files")
    ap.add_argument("--glob-b", default=None, help="Optional glob for group B JSON files")
    ap.add_argument("--metric", default="final_metrics.top_1_acc",
                    help="Nested metric path, e.g. final_metrics.top_1_acc")
    ap.add_argument("--output", default=None, help="Optional JSON output path")
    args = ap.parse_args()

    rec_a = load_records(args.glob_a, args.metric)
    vals_a = [r.value for r in rec_a]
    out = {
        "metric": args.metric,
        "group_a": {
            "glob": args.glob_a,
            "summary": summarize(vals_a),
            "records": [{"seed": r.seed, "value": r.value, "file": os.path.basename(r.path)} for r in rec_a],
        },
    }

    print(f"[Group A] {args.glob_a}")
    print(f"  n={out['group_a']['summary']['n']} "
          f"mean={out['group_a']['summary']['mean']:.6f} "
          f"std={out['group_a']['summary']['std']:.6f} "
          f"95%CI=±{out['group_a']['summary']['ci95']:.6f}")

    if args.glob_b:
        rec_b = load_records(args.glob_b, args.metric)
        vals_b = [r.value for r in rec_b]
        out["group_b"] = {
            "glob": args.glob_b,
            "summary": summarize(vals_b),
            "records": [{"seed": r.seed, "value": r.value, "file": os.path.basename(r.path)} for r in rec_b],
        }
        print(f"[Group B] {args.glob_b}")
        print(f"  n={out['group_b']['summary']['n']} "
              f"mean={out['group_b']['summary']['mean']:.6f} "
              f"std={out['group_b']['summary']['std']:.6f} "
              f"95%CI=±{out['group_b']['summary']['ci95']:.6f}")

        pairs, seeds = pair_by_seed(rec_a, rec_b)
        if not pairs:
            print("No paired seeds found between groups; Wilcoxon skipped.")
            out["wilcoxon"] = None
        else:
            x = [pa.value for pa, _ in pairs]
            y = [pb.value for _, pb in pairs]
            test = wilcoxon_signed_rank(x, y)
            diff = [a - b for a, b in zip(x, y)]
            out["wilcoxon"] = {
                "paired_seeds": seeds,
                "n_pairs": len(pairs),
                "mean_diff_a_minus_b": sum(diff) / len(diff),
                **test,
            }
            print(f"[Paired Wilcoxon] seeds={seeds}")
            print(f"  n_pairs={len(pairs)} mean_diff(A-B)={out['wilcoxon']['mean_diff_a_minus_b']:.6f}")
            print(f"  statistic={out['wilcoxon']['statistic']:.6f} p={out['wilcoxon']['pvalue']:.6g} ({out['wilcoxon']['method']})")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

