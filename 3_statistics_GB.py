#!/usr/bin/env python3
"""
Mixed-effects analysis of IF proportion (long-run, 1980–2019).

Main model:
    IF_proportion ~ Year_z + Year_z2 + number
      + (1 + Year_z + Year_z2 | target)

Add-on robustness check (for RQ1 / random intercept necessity):
    ML LRT comparing:
      Full:    (1 + Year_z + Year_z2 | target)
      Reduced: (0 + Year_z + Year_z2 | target)

- Fits ONE mixed-effects model across all targets.
- Extracts:
    * Fixed effects (for H1, H2, H3) + 95% CIs
    * Random effects per target (for RQ1, RQ2)
    * Derived intercepts, slopes, curvature per target (plural/singular)
    * Predicted plural & singular trajectories with 95% CIs (for plotting)
    * LRT for random intercept (saved to CSV + TXT)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# For the approximate likelihood-ratio test p-value
from scipy import stats


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def zscore(x: pd.Series) -> pd.Series:
    """Z-score a pandas Series using population SD (ddof=0)."""
    return (x - x.mean()) / x.std(ddof=0)


def ensure_number_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure 'number' is categorical with 'plural' as reference,
    so number[T.singular] compares singular vs plural.
    """
    if "number" not in df.columns:
        raise ValueError("Column 'number' not found in dataframe.")

    df = df.copy()
    df["number"] = df["number"].astype("category")

    cats = list(df["number"].cat.categories)
    if "plural" in cats and "singular" in cats:
        df["number"] = df["number"].cat.reorder_categories(
            ["plural", "singular"], ordered=True
        )
    return df


def predict_with_ci(fe: pd.Series, cov: pd.DataFrame, design: pd.Series):
    """
    Compute prediction and 95% CI for a linear combination X * beta.

    fe:     fixed-effects Series (indexed by parameter name)
    cov:    covariance matrix of fixed effects (DataFrame)
    design: Series giving the design row (same index as fe; 0 where term not used)

    Returns: (pred, se, ci_low, ci_high)
    """
    design = design.reindex(fe.index).fillna(0.0)

    pred = float((design * fe).sum())
    v = float(design.to_numpy() @ cov.to_numpy() @ design.to_numpy())
    se = np.sqrt(v) if v >= 0 else np.nan
    ci_low = pred - 1.96 * se
    ci_high = pred + 1.96 * se
    return pred, se, ci_low, ci_high


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Mixed-effects model of IF proportion (long-run)."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="output_GoogleBooks/identity_first_proportion_by_target.csv",
        help="Input CSV with columns: target, Year, number, IF, PF, IF_proportion",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="output_GoogleBooks/mixed_effects_long_run",
        help="Output directory for model summaries and parameter tables.",
    )
    parser.add_argument(
        "--min_year",
        type=int,
        default=1980,
        help="Minimum year for long-run analysis (default: 1980).",
    )
    parser.add_argument(
        "--max_year",
        type=int,
        default=2019,
        help="Maximum year for long-run analysis (default: 2019).",
    )
    args = parser.parse_args()

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------------- Load & filter data ----------------
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df):,} rows from {args.input_csv}")

    required_cols = {"target", "Year", "number", "IF_proportion"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    df = df[(df["Year"] >= args.min_year) & (df["Year"] <= args.max_year)].copy()
    if df.empty:
        raise ValueError(
            f"No data in the specified year range {args.min_year}–{args.max_year}"
        )

    df["Year"] = df["Year"].astype(int)
    df = ensure_number_categorical(df)

    # Z-score year within this window and create quadratic term
    df["Year_z"] = zscore(df["Year"])
    df["Year_z2"] = df["Year_z"] ** 2

    # Store training mean/sd for consistent prediction scaling
    year_mean = float(df["Year"].mean())
    year_sd = float(df["Year"].std(ddof=0))
    if year_sd == 0:
        raise ValueError("Year SD is zero after filtering; cannot standardize Year.")

    df = df.dropna(subset=["IF_proportion"]).copy()
    print(f"After filtering & NA-removal: {len(df):,} rows remain.")

    # ---------------- Fit mixed-effects model (REML) ----------------
    print("\nFitting mixed-effects model (REML):")
    print("  IF_proportion ~ Year_z + Year_z2 + number")
    print("  Random effects: (1 + Year_z + Year_z2 | target)")

    model = smf.mixedlm(
        "IF_proportion ~ Year_z + Year_z2 + number",
        data=df,
        groups=df["target"],
        re_formula="~ Year_z + Year_z2",
    )
    result = model.fit(method="lbfgs")
    print(result.summary())

    summary_path = outdir / "mixed_model_summary.txt"
    with open(summary_path, "w") as f:
        f.write(result.summary().as_text())
    print(f"\nSaved model summary -> {summary_path}")

    # ---------------- Robustness check: ML LRT for random intercept ----------------
    # NOTE: This is an approximate test; variance components are on the boundary.
    lrt_path = outdir / "random_intercept_lrt.csv"
    lrt_txt_path = outdir / "random_intercept_lrt.txt"

    print("\nRobustness check (ML LRT): random intercept vs no random intercept")
    lrt_rows = []

    try:
        # Full model (random intercept + slopes), ML
        model_full = smf.mixedlm(
            "IF_proportion ~ Year_z + Year_z2 + number",
            data=df,
            groups=df["target"],
            re_formula="~ Year_z + Year_z2",
        )
        res_full_ml = model_full.fit(method="lbfgs", reml=False)

        # Reduced model (random slopes only, no random intercept), ML
        # re_formula="0 + ..." removes the intercept from the random-effects design
        model_red = smf.mixedlm(
            "IF_proportion ~ Year_z + Year_z2 + number",
            data=df,
            groups=df["target"],
            re_formula="0 + Year_z + Year_z2",
        )
        res_red_ml = model_red.fit(method="lbfgs", reml=False)

        ll_full = float(res_full_ml.llf)
        ll_red = float(res_red_ml.llf)
        lr_stat = 2.0 * (ll_full - ll_red)
        df_diff = 1  # adding a single variance component (random intercept var)
        p_value = float(stats.chi2.sf(lr_stat, df=df_diff))

        lrt_rows.append(
            {
                "model_full": "(1 + Year_z + Year_z2 | target)",
                "model_reduced": "(0 + Year_z + Year_z2 | target)",
                "ll_full": ll_full,
                "ll_reduced": ll_red,
                "lr_stat": lr_stat,
                "df": df_diff,
                "p_value_chi2_approx": p_value,
            }
        )

        msg = (
            f"ML LRT (approx): LR={lr_stat:.3f}, df={df_diff}, "
            f"p={p_value:.4g}\n"
            f"  ll_full={ll_full:.3f}\n"
            f"  ll_reduced={ll_red:.3f}\n"
            "Note: This chi-square reference is approximate because variance components "
            "are tested on the boundary (0)."
        )
        print(msg)

        with open(lrt_txt_path, "w") as f:
            f.write(msg + "\n")
        print(f"Saved LRT text -> {lrt_txt_path}")

    except Exception as e:
        err_msg = f"Random-intercept LRT failed: {repr(e)}"
        print(err_msg)
        lrt_rows.append(
            {
                "model_full": "(1 + Year_z + Year_z2 | target)",
                "model_reduced": "(0 + Year_z + Year_z2 | target)",
                "ll_full": np.nan,
                "ll_reduced": np.nan,
                "lr_stat": np.nan,
                "df": 1,
                "p_value_chi2_approx": np.nan,
                "error": err_msg,
            }
        )
        with open(lrt_txt_path, "w") as f:
            f.write(err_msg + "\n")
        print(f"Saved LRT error -> {lrt_txt_path}")

    pd.DataFrame(lrt_rows).to_csv(lrt_path, index=False)
    print(f"Saved LRT table -> {lrt_path}")

    # ---------------- Fixed effects table (with CI) ----------------
    fe = result.params
    fe_se = result.bse
    fe_ci = result.conf_int()
    fe_df = pd.DataFrame(
        {
            "term": fe.index,
            "estimate": fe.values,
            "se": fe_se.values,
            "ci_low": fe_ci[0].values,
            "ci_high": fe_ci[1].values,
        }
    )
    fe_path = outdir / "fixed_effects.csv"
    fe_df.to_csv(fe_path, index=False)
    print(f"Saved fixed effects -> {fe_path}")

    # ---------------- Random effects per target ----------------
    re_dict = result.random_effects
    re_rows = []
    for target, re_params in re_dict.items():
        re_rows.append(
            {
                "target": target,
                "re_Intercept": re_params.get("Intercept", 0.0),
                "re_Year_z": re_params.get("Year_z", 0.0),
                "re_Year_z2": re_params.get("Year_z2", 0.0),
            }
        )

    re_df = pd.DataFrame(re_rows).sort_values("target")
    re_path = outdir / "random_effects_by_target.csv"
    re_df.to_csv(re_path, index=False)
    print(f"Saved random effects by target -> {re_path}")

    # ---------------- Derived per-target effects ----------------
    beta0 = fe.get("Intercept", np.nan)
    beta1 = fe.get("Year_z", np.nan)
    beta2 = fe.get("Year_z2", np.nan)
    beta3 = fe.get("number[T.singular]", 0.0)

    derived_rows = []
    for _, row in re_df.iterrows():
        t = row["target"]
        b0 = row["re_Intercept"]
        b1 = row["re_Year_z"]
        b2 = row["re_Year_z2"]

        intercept_plural = beta0 + b0
        slope_plural = beta1 + b1
        curvature_plural = beta2 + b2

        intercept_singular = beta0 + beta3 + b0
        slope_singular = slope_plural
        curvature_singular = curvature_plural

        derived_rows.append(
            {
                "target": t,
                "intercept_plural": intercept_plural,
                "intercept_singular": intercept_singular,
                "slope_plural": slope_plural,
                "slope_singular": slope_singular,
                "curvature_plural": curvature_plural,
                "curvature_singular": curvature_singular,
            }
        )

    derived_df = pd.DataFrame(derived_rows).sort_values("target")
    derived_path = outdir / "derived_effects_by_target.csv"
    derived_df.to_csv(derived_path, index=False)
    print(f"Saved derived per-target effects -> {derived_path}")

    # ---------------- Predicted plural & singular trajectories with CI ----------------
    cov_fe = result.cov_params()

    years = np.arange(args.min_year, args.max_year + 1)
    year_z = (pd.Series(years) - year_mean) / year_sd
    year_z2 = year_z ** 2

    rows = []
    for y, z, z2 in zip(years, year_z, year_z2):
        design_plural = pd.Series(0.0, index=fe.index)
        design_plural["Intercept"] = 1.0
        if "Year_z" in design_plural.index:
            design_plural["Year_z"] = float(z)
        if "Year_z2" in design_plural.index:
            design_plural["Year_z2"] = float(z2)

        design_singular = design_plural.copy()
        if "number[T.singular]" in design_singular.index:
            design_singular["number[T.singular]"] = 1.0

        p_pl, se_pl, lo_pl, hi_pl = predict_with_ci(fe, cov_fe, design_plural)
        p_sg, se_sg, lo_sg, hi_sg = predict_with_ci(fe, cov_fe, design_singular)

        rows.append(
            {
                "Year": int(y),
                "pred_plural": p_pl,
                "se_plural": se_pl,
                "ci_low_plural": lo_pl,
                "ci_high_plural": hi_pl,
                "pred_singular": p_sg,
                "se_singular": se_sg,
                "ci_low_singular": lo_sg,
                "ci_high_singular": hi_sg,
            }
        )

    traj_df = pd.DataFrame(rows)
    traj_path = outdir / "predicted_trajectories_with_ci.csv"
    traj_df.to_csv(traj_path, index=False)
    print(f"Saved predicted trajectories with 95% CI -> {traj_path}")

    print("\nMixed-effects analysis complete.")


if __name__ == "__main__":
    main()
