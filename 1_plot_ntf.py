#!/usr/bin/env python3
# Authors: Naomi Baes & ChatGPT
# Purpose: Plot Identity-First (IF) vs Person-First (PF) trends from long Ngram CSV.

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
import matplotlib.pyplot as plt


def load_long_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Year", "target", "form_type", "term", "ntf", "control"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {missing}")
    return df


def prepare_data(df: pd.DataFrame, per_million: bool, rolling: int) -> pd.DataFrame:
    # number: singular/plural  | form_class: IF/PF
    df = df.copy()
    df["number"] = df["form_type"].apply(lambda x: "singular" if "sg" in x else "plural")
    df["form_class"] = df["form_type"].apply(lambda x: "IF" if x.startswith("IF") else "PF")

    # scale to per million if requested
    if per_million:
        df["value"] = df["ntf"] * 1_000_000.0
    else:
        df["value"] = df["ntf"]

    # aggregate per Year/number/form_class (summing variants)
    agg = (
        df.groupby(["Year", "number", "form_class"], as_index=False)["value"]
          .sum()
          .pivot(index=["Year", "number"], columns="form_class", values="value")
          .reset_index()
          .fillna(0.0)
    )

    # optional rolling average per group (number)
    if rolling and rolling > 1:
        agg = (
            agg.sort_values(["number", "Year"])
               .groupby("number", group_keys=False)
               .apply(lambda g: g.assign(
                   IF=g["IF"].rolling(rolling, center=True, min_periods=1).mean(),
                   PF=g["PF"].rolling(rolling, center=True, min_periods=1).mean()
               ))
               .reset_index(drop=True)
        )
    return agg


def plot_if_pf_overall(agg: pd.DataFrame, outdir: Path, per_million: bool, rolling: int):
    ylab = "Occurrences per million words" if per_million else "Relative frequency (proportion)"
    for num in ["singular", "plural"]:
        g = agg[agg["number"] == num].sort_values("Year")
        if g.empty:
            continue
        plt.figure(figsize=(9, 5))
        plt.plot(g["Year"], g.get("IF", 0), label="IF (identity-first)")
        plt.plot(g["Year"], g.get("PF", 0), label="PF (person-first)")
        plt.title(
            f"IF vs PF over time — {num} "
            f"{'(rolling=' + str(rolling) + ')' if rolling and rolling > 1 else ''}"
        )
        plt.xlabel("Year")
        plt.ylabel(ylab)
        plt.grid(alpha=0.3, axis="both")
        plt.legend()
        outpath = outdir / f"IF_PF_{num}.png"
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()


def plot_per_target(
    df: pd.DataFrame,
    outdir: Path,
    per_million: bool,
    rolling: int,
    targets: Optional[List[str]] = None,
):
    ylab = "Occurrences per million words" if per_million else "Relative frequency (proportion)"
    tdir = outdir / "targets_ntf"
    tdir.mkdir(parents=True, exist_ok=True)

    # Precompute per target aggregation
    df = df.copy()
    df["number"] = df["form_type"].apply(lambda x: "singular" if "sg" in x else "plural")
    df["form_class"] = df["form_type"].apply(lambda x: "IF" if x.startswith("IF") else "PF")
    df["value"] = df["ntf"] * (1_000_000.0 if per_million else 1.0)

    if targets:
        df = df[df["target"].isin(targets)]

    for tgt, g0 in df.groupby("target"):
        # sum variants per Year/number/form_class
        g = (
            g0.groupby(["Year", "number", "form_class"], as_index=False)["value"]
              .sum()
              .pivot(index=["Year", "number"], columns="form_class", values="value")
              .reset_index()
              .fillna(0.0)
        )
        # rolling
        if rolling and rolling > 1:
            g = (
                g.sort_values(["number", "Year"])
                 .groupby("number", group_keys=False)
                 .apply(
                     lambda x: x.assign(
                         IF=x["IF"].rolling(rolling, center=True, min_periods=1).mean(),
                         PF=x["PF"].rolling(rolling, center=True, min_periods=1).mean(),
                     )
                 )
                 .reset_index(drop=True)
            )

        for num in ["singular", "plural"]:
            gnum = g[g["number"] == num].sort_values("Year")
            if gnum.empty:
                continue
            plt.figure(figsize=(9, 5))
            plt.plot(gnum["Year"], gnum.get("IF", 0), label="IF (identity-first)")
            plt.plot(gnum["Year"], gnum.get("PF", 0), label="PF (person-first)")
            plt.title(
                f"{tgt} — {num} "
                f"{'(rolling=' + str(rolling) + ')' if rolling and rolling > 1 else ''}"
            )
            plt.xlabel("Year")
            plt.ylabel(ylab)
            plt.grid(alpha=0.3, axis="both")
            plt.legend()
            outpath = tdir / f"{tgt}_IF_PF_{num}.png"
            plt.tight_layout()
            plt.savefig(outpath, dpi=200)
            plt.close()


def main():
    p = argparse.ArgumentParser(
        description="Plot Identity-First vs Person-First trends from long CSV."
    )
    p.add_argument(
        "--input_long_csv",
        type=str,
        required=True,
        help="Path to long CSV with columns: Year,target,form_type,term,ntf,control",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="output/plots",
        help="Directory to save PNGs",
    )
    p.add_argument(
        "--per_million",
        action="store_true",
        help="Plot per million words instead of proportions",
    )
    p.add_argument(
        "--rolling",
        type=int,
        default=0,
        help="Centered rolling window size (e.g., 7). 0 = no smoothing",
    )
    p.add_argument(
        "--targets",
        type=str,
        default="",
        help="Comma-separated list of targets to plot individually (default=all)",
    )
    p.add_argument(
        "--min_year",
        type=int,
        default=None,
        help="Minimum year to include in plots (e.g., 1980). Default: no lower bound.",
    )
    p.add_argument(
        "--max_year",
        type=int,
        default=None,
        help="Maximum year to include in plots (e.g., 2019). Default: no upper bound.",
    )
    args = p.parse_args()

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_long_csv(args.input_long_csv)

    # Filter by year range if specified
    if args.min_year is not None:
        df = df[df["Year"] >= args.min_year]
    if args.max_year is not None:
        df = df[df["Year"] <= args.max_year]

    if df.empty:
        raise ValueError(
            "No data left after applying year filters. "
            "Check --min_year and --max_year."
        )

    # Overall aggregated IF/PF plots (singular/plural)
    agg = prepare_data(df, per_million=args.per_million, rolling=args.rolling)
    plot_if_pf_overall(agg, outdir, per_million=args.per_million, rolling=args.rolling)

    # Per-target plots
    targets_list = (
        [t.strip() for t in args.targets.split(",") if t.strip()]
        if args.targets
        else None
    )
    plot_per_target(
        df,
        outdir,
        per_million=args.per_million,
        rolling=args.rolling,
        targets=targets_list,
    )

    # Also write out the aggregated CSV for reference
    agg_csv = outdir / "if_pf_aggregated.csv"
    agg.to_csv(agg_csv, index=False)
    print(f"Saved overall plots + per-target plots to: {outdir}")
    print(f"Saved aggregated CSV: {agg_csv}")


if __name__ == "__main__":
    main()
