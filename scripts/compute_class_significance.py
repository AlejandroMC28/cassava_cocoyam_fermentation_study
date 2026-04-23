#!/usr/bin/env python3
"""
Class-Level Chemical Significance
==================================
Computes Mann-Whitney U + BH-FDR significance for chemical-class
enrichment in fermented vs. control samples, across both cassava and cocoyam
and across every SIRIUS classification level (NPC pathway/superclass/class and
ClassyFire superclass/class/subclass/level 5).

For each (tuber, level, class) bucket:
  - Split per-feature log2FC into "in class" vs. "background" (all other
    features with a valid log2FC).
  - Mann-Whitney U (two-sided) on the two vectors.
  - Benjamini-Hochberg FDR correction within each (tuber, level) group.
  - Median per-feature log2FC of in-class features as the effect size.

This script produces the two CSVs used by figure2 and the
supplemental forest plot:

  results/canopus_npc_significance.csv     (NPC pathway/superclass/class)
  results/classyfire_significance.csv      (CF superclass/class/subclass/level 5)

Both CSVs share the schema:
  tuber,level,name,n,log2fc,adjp

Usage:
  python scripts/compute_class_significance.py
  python scripts/compute_class_significance.py --prob-cutoff 0.7 --min-class-n 5
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


# ============================================================================
# CONFIGURATION
# ============================================================================
TUBERS = [
    # (batch_slug, short_slug, display_name)
    ("cassavaonly", "cassava", "Cassava"),
    ("cocoyamonly", "cocoyam", "Cocoyam"),
]

NPC_LEVELS = [
    ("pathway", "sirius_NPC#pathway", "sirius_NPC#pathway Probability"),
    ("superclass", "sirius_NPC#superclass", "sirius_NPC#superclass Probability"),
    ("class", "sirius_NPC#class", "sirius_NPC#class Probability"),
]

CF_LEVELS = [
    (
        "superclass",
        "sirius_ClassyFire#superclass",
        "sirius_ClassyFire#superclass probability",
    ),
    ("class", "sirius_ClassyFire#class", "sirius_ClassyFire#class Probability"),
    (
        "subclass",
        "sirius_ClassyFire#subclass",
        "sirius_ClassyFire#subclass Probability",
    ),
    ("level 5", "sirius_ClassyFire#level 5", "sirius_ClassyFire#level 5 Probability"),
]

# Classes figure2 hardcodes for panels c/d. Emit a warning if any fall below
# the probability/min_class_n filter and end up absent from the output.
FIGURE2_SELECTED_CLASSES = {
    "Cassava": [
        "Diacylglycerols",
        "Glycerophosphocholines",
        "Dipeptides",
        "N-acyl amines",
        "Pinane monoterpenoids",
        "Cholestane steroids",
    ],
    "Cocoyam": [
        "N-acyl amines",
        "Amino acids and Peptides",
        "Cyanogenic glycosides",
        "Polysaccharides",
        "Cyclic peptides",
        "Tripeptides",
    ],
}


# ============================================================================
# CORE TEST
# ============================================================================
def canopus_style_test(merged, class_col, prob_col, *, prob_cutoff, min_class_n):
    """Mann-Whitney U on in-class vs. background per-feature log2FC.

    Returns a DataFrame with one row per class meeting min_class_n, columns:
    name, n, log2fc (median in-class log2FC), pval (raw, pre-FDR).
    """
    prob = pd.to_numeric(merged[prob_col], errors="coerce")
    labeled = merged[
        (prob >= prob_cutoff) & merged[class_col].notna() & (merged[class_col] != "")
    ]

    all_fc = pd.to_numeric(merged["log2FC"], errors="coerce")

    rows = []
    for cls, grp in labeled.groupby(class_col):
        in_ids = grp["id"]
        in_fc = all_fc.loc[merged["id"].isin(in_ids)].dropna().values
        out_fc = all_fc.loc[~merged["id"].isin(in_ids)].dropna().values
        if len(in_fc) < min_class_n:
            continue
        pval = mannwhitneyu(
            in_fc, out_fc, alternative="two-sided", method="auto"
        ).pvalue
        rows.append(
            {
                "name": cls,
                "n": int(len(in_fc)),
                "log2fc": float(np.mean(in_fc)),
                "pval": float(pval),
            }
        )
    return pd.DataFrame(rows, columns=["name", "n", "log2fc", "pval"])


def bh_adjust(df):
    """Return df with an 'adjp' column (BH-FDR) replacing 'pval'."""
    if df.empty:
        out = df.drop(columns=["pval"]).copy()
        out["adjp"] = pd.Series(dtype=float)
        return out
    _, adj, _, _ = multipletests(df["pval"].values, method="fdr_bh")
    out = df.drop(columns=["pval"]).copy()
    out["adjp"] = adj
    return out


# ============================================================================
# PER-TUBER PIPELINE
# ============================================================================
def load_merged(project_dir, batch_slug, short_slug):
    """Left-join FDR log2FC onto MS2 annotations on feature id."""
    batch = f"b3_{batch_slug}"
    fdr_path = os.path.join(
        project_dir,
        "results",
        f"{batch}_fdr_{short_slug}_vs_fermented_{short_slug}.csv",
    )
    ms2_path = os.path.join(project_dir, "results", f"{batch}_ms2_annotations.csv")

    fdr = pd.read_csv(fdr_path)
    ms2 = pd.read_csv(ms2_path).rename(columns={"mappingFeatureId": "id"})
    merged = fdr.merge(ms2, on="id", how="left")
    assert merged["id"].is_unique, (
        f"Duplicate feature ids after merge for {batch_slug} "
        f"— check ms2_annotations for duplicate mappingFeatureId rows."
    )
    return merged


def run_levels(merged, tuber_name, level_specs, prob_cutoff, min_class_n):
    """Run canopus_style_test for each level and collect tagged rows."""
    tagged = []
    for level_label, class_col, prob_col in level_specs:
        if class_col not in merged.columns or prob_col not in merged.columns:
            print(
                f"  [warn] {tuber_name} / {level_label}: missing columns "
                f"({class_col!r} or {prob_col!r}), skipped.",
                file=sys.stderr,
            )
            continue
        res = canopus_style_test(
            merged,
            class_col,
            prob_col,
            prob_cutoff=prob_cutoff,
            min_class_n=min_class_n,
        )
        res = bh_adjust(res)
        n_sig = int((res["adjp"] < 0.05).sum()) if not res.empty else 0
        print(f"  {level_label:<11} tested={len(res):>4}  sig(adjp<0.05)={n_sig}")
        res.insert(0, "level", level_label)
        res.insert(0, "tuber", tuber_name)
        tagged.append(res)
    if not tagged:
        return pd.DataFrame(columns=["tuber", "level", "name", "n", "log2fc", "adjp"])
    return pd.concat(tagged, ignore_index=True)


# ============================================================================
# ENTRY POINT
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute Mann-Whitney U + BH-FDR significance for "
            "chemical classes across cassava and cocoyam."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--project-dir",
        default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        help="Project root (default: parent of scripts/).",
    )
    parser.add_argument(
        "--prob-cutoff",
        type=float,
        default=0.64,
        help="SIRIUS classification probability cutoff (default: 0.64).",
    )
    parser.add_argument(
        "--min-class-n",
        type=int,
        default=3,
        help="Minimum features per class to run the test (default: 3).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for CSVs (default: <project-dir>/results).",
    )
    args = parser.parse_args()

    project_dir = os.path.expanduser(args.project_dir)
    out_dir = args.out_dir or os.path.join(project_dir, "results")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 72)
    print("Class-level significance — Mann-Whitney U + BH-FDR")
    print(f"  project-dir  : {project_dir}")
    print(f"  prob-cutoff  : {args.prob_cutoff}")
    print(f"  min-class-n  : {args.min_class_n}")
    print("=" * 72)

    npc_tables = []
    cf_tables = []
    for batch_slug, short_slug, display_name in TUBERS:
        print(f"\n[{display_name}]  loading {batch_slug}…")
        merged = load_merged(project_dir, batch_slug, short_slug)
        print(
            f"  merged features: {len(merged):,}  (with log2FC: {merged['log2FC'].notna().sum():,})"
        )

        print(f"  NPC:")
        npc = run_levels(
            merged, display_name, NPC_LEVELS, args.prob_cutoff, args.min_class_n
        )
        print(f"  ClassyFire:")
        cf = run_levels(
            merged, display_name, CF_LEVELS, args.prob_cutoff, args.min_class_n
        )

        # Flag missing SELECTED_CLASSES (only relevant at NPC class level)
        npc_class = npc[npc["level"] == "class"]
        missing = [
            name
            for name in FIGURE2_SELECTED_CLASSES.get(display_name, [])
            if name not in set(npc_class["name"])
        ]
        if missing:
            print(
                f"  [warn] {display_name}: SELECTED_CLASSES missing from NPC class output: "
                f"{missing}  — figure2 panels c/d will fail its length assertion.",
                file=sys.stderr,
            )

        npc_tables.append(npc)
        cf_tables.append(cf)

    npc_all = pd.concat(npc_tables, ignore_index=True)
    cf_all = pd.concat(cf_tables, ignore_index=True)

    # Column order and rounding to match existing CSV presentation
    npc_all = npc_all[["tuber", "level", "name", "n", "log2fc", "adjp"]]
    cf_all = cf_all[["tuber", "level", "name", "n", "log2fc", "adjp"]]
    npc_all["log2fc"] = npc_all["log2fc"].round(4)
    npc_all["adjp"] = npc_all["adjp"].round(6)
    cf_all["log2fc"] = cf_all["log2fc"].round(4)
    cf_all["adjp"] = cf_all["adjp"].round(6)

    npc_all = npc_all.sort_values(["tuber", "level", "adjp"]).reset_index(drop=True)
    cf_all = cf_all.sort_values(["tuber", "level", "adjp"]).reset_index(drop=True)

    npc_path = os.path.join(out_dir, "canopus_npc_significance.csv")
    cf_path = os.path.join(out_dir, "classyfire_significance.csv")
    npc_all.to_csv(npc_path, index=False)
    cf_all.to_csv(cf_path, index=False)

    print("\n" + "=" * 72)
    print(f"Wrote {len(npc_all):>4} rows -> {npc_path}")
    print(f"Wrote {len(cf_all):>4} rows -> {cf_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
