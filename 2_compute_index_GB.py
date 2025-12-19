#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    # ------------------------------
    # Parse CLI arguments
    # ------------------------------
    parser = argparse.ArgumentParser(
        description="Compute IF vs PF proportions from long-format Ngram CSV."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to output_GoogleBooks/ngram_relative_long.csv"
    )
    parser.add_argument(
        "--min_year",
        type=int,
        default=None,
        help="Minimum year to include (e.g., 1980). Default: no lower bound."
    )
    parser.add_argument(
        "--max_year",
        type=int,
        default=None,
        help="Maximum year to include (e.g., 2019). Default: no upper bound."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="output_GoogleBooks/identity_first_proportion_by_target.csv",
        help="Where to save the per-target output CSV"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save overall output CSVs. Default: same directory as --output_csv."
    )
    args = parser.parse_args()

    out_by_target_path = Path(args.output_csv)
    out_dir = Path(args.output_dir) if args.output_dir else out_by_target_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------
    # Load data
    # ------------------------------
    df = pd.read_csv(args.input_csv)

    expected_cols = {"Year", "form_type", "ntf", "target"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")

    # ------------------------------
    # Apply year filtering
    # ------------------------------
    if args.min_year is not None:
        df = df[df["Year"] >= args.min_year]

    if args.max_year is not None:
        df = df[df["Year"] <= args.max_year]

    if df.empty:
        raise ValueError("No data left after applying year filters.")

    # ------------------------------
    # Derive number and form_class
    # ------------------------------
    df["number"] = df["form_type"].apply(lambda x: "singular" if "sg" in str(x) else "plural")
    df["form_class"] = df["form_type"].apply(lambda x: "IF" if str(x).startswith("IF") else "PF")

    # ------------------------------
    # Aggregate per target / year / number / form_class
    # ------------------------------
    agg = (
        df.groupby(["target", "Year", "number", "form_class"], as_index=False)["ntf"]
          .sum()
    )

    # ------------------------------
    # Pivot wider: IF and PF as columns
    # ------------------------------
    agg_wide = (
        agg.pivot(
            index=["target", "Year", "number"],
            columns="form_class",
            values="ntf"
        )
        .fillna(0)
        .reset_index()
    )

    # Ensure both columns exist
    if "IF" not in agg_wide.columns:
        agg_wide["IF"] = 0.0
    if "PF" not in agg_wide.columns:
        agg_wide["PF"] = 0.0

    # ------------------------------
    # Compute per-target IF proportion
    # ------------------------------
    denom = agg_wide["IF"] + agg_wide["PF"]
    agg_wide["IF_proportion"] = np.where(denom > 0, agg_wide["IF"] / denom, np.nan)

    # ------------------------------
    # Save per-target output (existing behavior)
    # ------------------------------
    result = agg_wide[["target", "Year", "number", "IF", "PF", "IF_proportion"]]
    result.to_csv(out_by_target_path, index=False)

    # ------------------------------
    # Compute OVERALL IF proportion (2 ways)
    # ------------------------------

    # 1) Unweighted mean across targets (mean of target-level proportions)
    overall_unweighted = (
        agg_wide
        .groupby(["Year", "number"], as_index=False)["IF_proportion"]
        .mean()
    )

    # 2) Weighted/pooled across targets (sum IF and PF then compute proportion)
    overall_weighted = (
        agg_wide
        .groupby(["Year", "number"], as_index=False)[["IF", "PF"]]
        .sum()
    )
    denom2 = overall_weighted["IF"] + overall_weighted["PF"]
    overall_weighted["IF_proportion"] = np.where(denom2 > 0, overall_weighted["IF"] / denom2, np.nan)
    overall_weighted = overall_weighted[["Year", "number", "IF", "PF", "IF_proportion"]]

    # ------------------------------
    # Save overall outputs
    # ------------------------------
    out_overall_unweighted = out_dir / "identity_first_proportion_overall_unweighted.csv"
    out_overall_weighted = out_dir / "identity_first_proportion_overall_weighted.csv"

    overall_unweighted.to_csv(out_overall_unweighted, index=False)
    overall_weighted.to_csv(out_overall_weighted, index=False)

    print(f"Saved per-target IF/PF proportion table → {out_by_target_path}")
    print(f"Saved OVERALL IF proportion (unweighted mean across targets) → {out_overall_unweighted}")
    print(f"Saved OVERALL IF proportion (weighted/pooled across targets) → {out_overall_weighted}")

if __name__ == "__main__":
    main()
