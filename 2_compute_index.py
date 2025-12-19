#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np

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
        help="Path to output/ngram_relative_long.csv"
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
        default="output/identity_first_proportion_by_target.csv",
        help="Where to save the output CSV"
    )
    args = parser.parse_args()

    # ------------------------------
    # Load data
    # ------------------------------
    df = pd.read_csv(args.input_csv)

    expected_cols = {"Year", "form_type", "ntf", "target"}
    missing = expected_cols - set(df.columns)
    assert not missing, f"Missing columns in df: {missing}"

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
    df["number"] = df["form_type"].apply(
        lambda x: "singular" if "sg" in x else "plural"
    )
    df["form_class"] = df["form_type"].apply(
        lambda x: "IF" if x.startswith("IF") else "PF"
    )

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
    # Compute IF proportion
    # ------------------------------
    denom = agg_wide["IF"] + agg_wide["PF"]
    agg_wide["IF_proportion"] = np.where(
        denom > 0,
        agg_wide["IF"] / denom,
        np.nan
    )

    # ------------------------------
    # Final output
    # ------------------------------
    result = agg_wide[["target", "Year", "number", "IF", "PF", "IF_proportion"]]
    result.to_csv(args.output_csv, index=False)

    print(f"Saved IF/PF proportion table → {args.output_csv}")


if __name__ == "__main__":
    main()
