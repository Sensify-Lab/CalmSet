#!/usr/bin/env python3
"""
Finalize gold-standard top-3 emotions per song using:
- Agreement-weighted Borda over workers' top-3 (agreement in [-2..2], weight = 1 + ALPHA * agreement, floored)
- CLAP-assisted tie-break, gated by mean per-song agreement >= --clap-agree-thresh
- Deterministic resolution (no randomness)
- Supports multiple worker rows per song; merges all ballots for a filename

Expected files (same directory as this script):
  mturk_compiled_final.csv  (workers; wide rows)
    columns (minimum used here):
      filename
      emotion1_w1, emotion2_w1, emotion3_w1
      emotion1_w2, emotion2_w2, emotion3_w2
      emotion1_w3, emotion2_w3, emotion3_w3
      agreement_w1, agreement_w2, agreement_w3   (integers in -2..2)

  clap_combined.csv  (CLAP predictions)
    columns:
      file_name  (will be normalized to 'filename')
      emotion1, emotion2, emotion3   (optional explicit CLAP top-3)
      and/or per-tag scores among:
        Stimulating, Playful, Soothing, Sensory-Calming,
        Grounding, Focusing, Transitional, Anxiety-Reduction

Outputs (in --out directory):
  human_aggregated.csv          (final top-3 for human-annotated songs, with diagnostics)
  final_gold_combined.csv       (all songs: human where available, CLAP-only otherwise)
  summary.json                  (key counts/percentages to paste in the paper)

Run examples:
  python finalize_gold_with_policy.py
  python finalize_gold_with_policy.py --clap-agree-thresh 1.0
  python finalize_gold_with_policy.py --workers_csv mturk_compiled_final.csv --clap_csv clap_combined.csv --out results_final_gold_policy
"""

import argparse
from collections import defaultdict, Counter
from pathlib import Path
import json
import numpy as np
import pandas as pd

# -----------------------------
# CONFIG (adjust if desired)
# -----------------------------
# Borda points per rank
BORDA_POINTS = {1: 3, 2: 2, 3: 1}

# Agreement weight: w = max(WEIGHT_FLOOR, 1 + ALPHA * agreement), where agreement in [-2..2]
ALPHA = 0.25
WEIGHT_FLOOR = 0.25  # ensures every worker's ballot still counts a bit

# Tag vocabulary (used to detect per-tag score columns)
KNOWN_TAGS = [
    "Stimulating", "Playful", "Soothing", "Sensory-Calming",
    "Grounding", "Focusing", "Transitional", "Anxiety-Reduction"
]

# -----------------------------
# Helpers
# -----------------------------
def _normalize_tag(x):
    return str(x).strip() if isinstance(x, str) else ""

def weight_from_agreement(a):
    if a is None or pd.isna(a):
        return 1.0
    return max(WEIGHT_FLOOR, 1.0 + ALPHA * float(a))  # -> [0.5..1.5] for ALPHA=0.25

def extract_clap_structures(clap_df: pd.DataFrame):
    """
    Returns:
      clap_order[filename]  -> list of up to 3 tags (CLAP rank order)
      clap_scores[filename] -> dict {tag: score} if available, else {}
    """
    df = clap_df.copy()
    if "file_name" in df.columns and "filename" not in df.columns:
        df = df.rename(columns={"file_name": "filename"})
    df["filename"] = df["filename"].astype(str).str.strip()

    has_explicit = {"emotion1", "emotion2", "emotion3"}.issubset(df.columns)
    score_cols = [c for c in df.columns if c in KNOWN_TAGS]

    clap_order, clap_scores = {}, {}
    for _, r in df.iterrows():
        fn = r["filename"]

        # explicit top-3 (if present)
        ordered = []
        if has_explicit:
            ordered = [_normalize_tag(r["emotion1"]),
                       _normalize_tag(r["emotion2"]),
                       _normalize_tag(r["emotion3"])]
            ordered = [t for t in ordered if t]

        # per-tag scores (if present)
        scores = {}
        for t in score_cols:
            try:
                scores[t] = float(r[t])
            except Exception:
                scores[t] = None

        # derive top-3 from scores if explicit not present
        if not ordered and score_cols:
            ranked = sorted(scores.items(), key=lambda kv: -(kv[1] if kv[1] is not None else -1e9))
            ordered = [k for k, _ in ranked[:3]]

        clap_order[fn] = ordered
        clap_scores[fn] = scores

    return clap_order, clap_scores

def per_row_ballots(row: pd.Series):
    """
    Extract up to 3 ranked lists + weights from one wide worker row.
    Duplicates within a worker are removed (first occurrence kept).
    """
    ranklists, weights = [], []
    for w in (1, 2, 3):
        e1, e2, e3 = row.get(f"emotion1_w{w}", None), row.get(f"emotion2_w{w}", None), row.get(f"emotion3_w{w}", None)
        rl = [_normalize_tag(e1), _normalize_tag(e2), _normalize_tag(e3)]
        rl = [t for t in rl if t]
        # enforce uniqueness within one worker ballot
        seen, unique_rl = set(), []
        for t in rl:
            if t not in seen:
                unique_rl.append(t); seen.add(t)
        if unique_rl:
            ranklists.append(unique_rl)
            weights.append(weight_from_agreement(row.get(f"agreement_w{w}", None)))
    return ranklists, weights

def aggregate_all_ballots(group_df: pd.DataFrame):
    """Merge ALL wide rows for a filename into one list of ballots + weights."""
    all_ranklists, all_weights = [], []
    for _, r in group_df.iterrows():
        rls, ws = per_row_ballots(r)
        all_ranklists.extend(rls)
        all_weights.extend(ws)
    return all_ranklists, all_weights

def borda_scores(ranklists, weights):
    scores = defaultdict(float)
    for rl, w in zip(ranklists, weights):
        for i, t in enumerate(rl, start=1):  # ranks 1..3
            scores[t] += w * BORDA_POINTS.get(i, 0)
    return scores

def gather_mean_agreement(group_df: pd.DataFrame):
    vals = []
    for _, r in group_df.iterrows():
        for w in (1, 2, 3):
            a = r.get(f"agreement_w{w}", None)
            if a is not None and not pd.isna(a):
                vals.append(float(a))
    return float(np.mean(vals)) if len(vals) else 0.0

def tie_break(candidates, clap_order, tag_freq=None, clap_scores=None):
    """
    Deterministic multi-step tie-break:
      1) CLAP order
      2) Higher CLAP per-tag score (if available)
      3) Higher worker frequency within the song
      4) Alphabetical (stable fallback)
    """
    cands = list(candidates)

    # 1) CLAP order
    if clap_order:
        in_order = [t for t in clap_order if t in cands]
        leftover = [t for t in cands if t not in in_order]
        cands = in_order + leftover

    # 2) CLAP score
    if clap_scores and len(cands) > 1:
        cands = sorted(
            cands,
            key=lambda t: -(clap_scores.get(t, -1e9) if clap_scores.get(t, None) is not None else -1e9)
        )

    # 3) Worker frequency
    if tag_freq and len(cands) > 1:
        cands = sorted(cands, key=lambda t: -tag_freq.get(t, 0))

    # 4) Alphabetical fallback
    if len(cands) > 1:
        cands = sorted(cands, key=lambda t: t)

    return cands

def finalize_with_gate(scores, clap_order, clap_scores, tag_freq, allow_clap: bool):
    """
    Convert score dict to final top-3 using tie-break rules.
    Only uses CLAP info if allow_clap=True (mean agreement >= threshold).
    """
    if not scores:
        return [], {"had_tie": False, "tie_level": 0, "used_clap": False}

    # group by score to detect ties
    buckets = defaultdict(list)
    for t, sc in scores.items():
        buckets[sc].append(t)
    sorted_scores = sorted(buckets.keys(), reverse=True)

    ordered, used_clap, tie_level = [], False, 0
    for sc in sorted_scores:
        cands = buckets[sc]
        if len(cands) == 1:
            ordered += cands
        else:
            tie_level = max(tie_level, len(ordered) + 1)
            before = cands[:]
            cands = tie_break(
                cands,
                clap_order if allow_clap else [],
                tag_freq,
                clap_scores if allow_clap else {}
            )
            used_clap = used_clap or (before != cands and allow_clap and bool(clap_order))
            ordered += cands
        if len(ordered) >= 3:
            break

    return ordered[:3], {"had_tie": tie_level > 0, "tie_level": tie_level, "used_clap": used_clap}

# -----------------------------
# Pipeline
# -----------------------------
def run_pipeline(workers_csv="mturk_compiled_final.csv",
                 clap_csv="clap_combined.csv",
                 out_dir="results_final_gold_policy",
                 clap_agree_thresh=0.0):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load
    workers = pd.read_csv(workers_csv)
    clap = pd.read_csv(clap_csv)

    # Normalize keys
    if "file_name" in clap.columns and "filename" not in clap.columns:
        clap = clap.rename(columns={"file_name": "filename"})
    workers["filename"] = workers["filename"].astype(str).str.strip()
    clap["filename"] = clap["filename"].astype(str).str.strip()

    # Build CLAP structures
    clap_order, clap_scores_map = extract_clap_structures(clap)

    # Aggregate per human song
    rows = []
    for fn, group in workers.groupby("filename"):
        ranklists, weights = aggregate_all_ballots(group)
        tag_freq = Counter([t for rl in ranklists for t in rl])
        scores = borda_scores(ranklists, weights)

        mean_agree = gather_mean_agreement(group)
        allow_clap = (mean_agree >= clap_agree_thresh)

        top3, tieinfo = finalize_with_gate(
            scores,
            clap_order.get(fn, []),
            clap_scores_map.get(fn, {}),
            tag_freq,
            allow_clap=allow_clap
        )

        clap_top3 = clap_order.get(fn, [])
        jacc = (len(set(top3) & set(clap_top3)) / len(set(top3) | set(clap_top3))
                if top3 and clap_top3 else 0.0)

        rows.append({
            "filename": fn,
            "final_top1": top3[0] if len(top3) > 0 else "",
            "final_top2": top3[1] if len(top3) > 1 else "",
            "final_top3": top3[2] if len(top3) > 2 else "",
            "clap_top1": clap_top3[0] if len(clap_top3) > 0 else "",
            "clap_top2": clap_top3[1] if len(clap_top3) > 1 else "",
            "clap_top3": clap_top3[2] if len(clap_top3) > 2 else "",
            "had_tie": tieinfo["had_tie"],
            "tie_level": tieinfo["tie_level"],
            "used_clap_tiebreak": tieinfo["used_clap"],
            "final_vs_clap_jaccard": jacc,
            "mean_agreement": mean_agree,
            "allow_clap": allow_clap
        })

    human = pd.DataFrame(rows).sort_values("filename")
    human_path = out / "human_aggregated.csv"
    human.to_csv(human_path, index=False)

    # Build combined (all songs): human where available, else CLAP-only
    human_set = set(human["filename"])
    if {"emotion1", "emotion2", "emotion3"}.issubset(clap.columns):
        clap_only = clap[~clap["filename"].isin(human_set)][["filename", "emotion1", "emotion2", "emotion3"]].copy()
        clap_only = clap_only.rename(columns={"emotion1": "final_top1",
                                              "emotion2": "final_top2",
                                              "emotion3": "final_top3"})
    else:
        score_cols = [c for c in clap.columns if c in KNOWN_TAGS]
        if not score_cols:
            raise ValueError("CLAP CSV must contain explicit top-3 or per-tag score columns.")
        def top3_from_scores(row):
            ranked = sorted([(t, row[t]) for t in score_cols],
                            key=lambda kv: -(kv[1] if pd.notna(kv[1]) else -1e9))
            tops = [t for t, _ in ranked[:3]]
            return pd.Series({
                "final_top1": tops[0] if len(tops) > 0 else "",
                "final_top2": tops[1] if len(tops) > 1 else "",
                "final_top3": tops[2] if len(tops) > 2 else "",
            })
        clap_only = clap[~clap["filename"].isin(human_set)].copy()
        clap_only[["final_top1", "final_top2", "final_top3"]] = clap_only.apply(top3_from_scores, axis=1)

    combined = pd.concat([
        human[["filename", "final_top1", "final_top2", "final_top3"]].assign(
            source=f"Human+Borda (agree-weighted, CLAP gated ≥{clap_agree_thresh})"
        ),
        clap_only[["filename", "final_top1", "final_top2", "final_top3"]].assign(source="CLAP_only"),
    ], ignore_index=True).sort_values("filename")

    combined_path = out / "final_gold_combined.csv"
    combined.to_csv(combined_path, index=False)

    # Summary for the paper
    summary = {
        "n_human_songs": int(human["filename"].nunique()),
        "ties_any_count": int(human["had_tie"].sum()),
        "ties_any_rate": float(human["had_tie"].mean()) if len(human) else 0.0,
        "ties_broken_with_clap_count": int(human["used_clap_tiebreak"].sum()),
        "ties_broken_with_clap_rate": float(human["used_clap_tiebreak"].mean()) if len(human) else 0.0,
        "mean_final_vs_clap_jaccard": float(human["final_vs_clap_jaccard"].mean()) if len(human) else 0.0,
        "median_final_vs_clap_jaccard": float(human["final_vs_clap_jaccard"].median()) if len(human) else 0.0,
        "mean_agreement_over_songs": float(human["mean_agreement"].mean()) if len(human) else 0.0,
        "clap_allowed_rate": float(human["allow_clap"].mean()) if len(human) else 0.0,
        "clap_agree_threshold": float(clap_agree_thresh),
    }
    summary_path = out / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n[Done] Wrote:")
    print(f"  - {human_path}")
    print(f"  - {combined_path}")
    print(f"  - {summary_path}\n")
    print("[Human subset summary]")
    for k, v in summary.items():
        print(f"{k}: {v}")

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Finalize gold with agreement-weighted Borda + CLAP gated by mean agreement.")
    ap.add_argument("--workers_csv", default="mturk_compiled_final.csv", help="Path to workers CSV")
    ap.add_argument("--clap_csv", default="clap_combined.csv", help="Path to CLAP CSV")
    ap.add_argument("--out", default="results_final_gold_policy", help="Output directory")
    ap.add_argument("--clap-agree-thresh", type=float, default=0.0,
                    help="Mean agreement threshold to allow CLAP tie-break (e.g., 0.0 or 1.0).")
    args = ap.parse_args()
    run_pipeline(args.workers_csv, args.clap_csv, args.out, clap_agree_thresh=args.clap_agree_thresh)

if __name__ == "__main__":
    main()
