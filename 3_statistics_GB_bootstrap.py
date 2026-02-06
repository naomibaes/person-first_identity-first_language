#!/usr/bin/env python3
"""
Mixed-effects analysis of IF proportion (long-run, 1980–2019).

Main model (REML):
    IF_proportion ~ Year_z + Year_z2 + number
      + (1 + Year_z + Year_z2 | target)

Robustness check (ML LRT for random intercept):
    Full:    (1 + Year_z + Year_z2 | target)
    Reduced: (0 + Year_z + Year_z2 | target)

Outputs:
    * Fixed effects + 95% CIs
    * Random effects per target
    * Derived intercepts/slopes/curvature per target
    * Predicted plural & singular trajectories with 95% CIs (fixed-effects-only)
    * LRT for random intercept
    * OPTIONAL: Parametric bootstrap CIs for per-target slopes/curvature
"""

import argparse
import subprocess
import warnings
from pathlib import Path
from numpy.random import default_rng

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.tools.sm_exceptions import ConvergenceWarning


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def zscore(x: pd.Series) -> pd.Series:
    sd = x.std(ddof=0)
    if sd == 0:
        raise ValueError("Cannot z-score: SD is zero.")
    return (x - x.mean()) / sd


def ensure_number_categorical(df: pd.DataFrame) -> pd.DataFrame:
    if "number" not in df.columns:
        raise ValueError("Column 'number' not found in dataframe.")
    df = df.copy()
    df["number"] = df["number"].astype("category")
    cats = list(df["number"].cat.categories)
    if "plural" in cats and "singular" in cats:
        df["number"] = df["number"].cat.reorder_categories(["plural", "singular"], ordered=True)
    return df


def get_git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "UNKNOWN"


def extract_condition_slopes_fast(fe_params, random_effects, targets):
    """Return arrays of per-target slope/curvature (fixed + random)."""
    b_year = float(fe_params.get("Year_z", 0.0))
    b_year2 = float(fe_params.get("Year_z2", 0.0))

    slopes = np.empty(len(targets), dtype=float)
    curvs = np.empty(len(targets), dtype=float)

    for i, t in enumerate(targets):
        re_t = random_effects[t]
        slopes[i] = b_year + float(re_t.get("Year_z", 0.0))
        curvs[i] = b_year2 + float(re_t.get("Year_z2", 0.0))

    return slopes, curvs


def fast_predict_ci(fe_vec: np.ndarray, cov: np.ndarray, x: np.ndarray):
    """Scalar prediction + Wald CI."""
    pred = float(x @ fe_vec)
    v = float(x @ cov @ x)
    se = np.sqrt(v) if v >= 0 else np.nan
    lo = pred - 1.96 * se
    hi = pred + 1.96 * se
    return pred, se, lo, hi


# ---------------------------------------------------------------------
# Bootstrap (faster + quieter)
# ---------------------------------------------------------------------

def parametric_bootstrap_slopes_fast(
    fitted_result,
    base_model,
    df: pd.DataFrame,
    group_col: str,
    nsim: int = 500,
    seed: int = 123,
    fit_method: str = "lbfgs",
    target_success: int | None = None,
    max_fail_frac: float = 0.90,
    maxiter: int = 120,
):
    rng = default_rng(seed)
    targets = sorted(df[group_col].unique().tolist())
    n_targets = len(targets)

    # Fixed-effects-only mean Xβ (NO BLUPs)
    mu_fixed = np.asarray(fitted_result.model.exog @ fitted_result.fe_params).ravel()

    resid_sd = float(np.sqrt(fitted_result.scale))
    cov_re = np.asarray(fitted_result.cov_re)
    Z = np.asarray(fitted_result.model.exog_re)

    # indices by group + pre-sliced Z blocks (saves time)
    group_vals = df[group_col].to_numpy()
    group_index = [np.where(group_vals == t)[0] for t in targets]
    Z_by_group = [Z[idx, :] for idx in group_index]

    k_re = cov_re.shape[0]

    # suppress convergence spam inside bootstrap
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    slope_draws = []
    curv_draws = []
    successes = 0
    failures = 0
    requested = nsim

    if target_success is None:
        target_success = nsim

    # Pre-allocate endog buffer to reduce reallocations
    y_star = np.empty_like(mu_fixed)

    for b in range(nsim):
        # draw all group REs at once: shape (n_targets, k_re)
        U = rng.multivariate_normal(mean=np.zeros(k_re), cov=cov_re, size=n_targets)

        # simulate outcome
        y_star[:] = mu_fixed
        for i in range(n_targets):
            y_star[group_index[i]] += Z_by_group[i] @ U[i]
        y_star += rng.normal(0.0, resid_sd, size=y_star.shape[0])

        try:
            base_model.endog = y_star  # reuse same model object
            r = base_model.fit(method=fit_method, disp=False, reml=False, maxiter=maxiter)

            slopes, curvs = extract_condition_slopes_fast(r.fe_params, r.random_effects, targets)
            slope_draws.append(slopes)
            curv_draws.append(curvs)
            successes += 1

        except Exception:
            failures += 1

        if successes >= target_success:
            break

        done = b + 1
        if done >= 50:
            fail_frac = failures / done
            if fail_frac >= max_fail_frac:
                raise RuntimeError(
                    f"Bootstrap aborting: too many failed refits ({failures}/{done}, {fail_frac:.2%}). "
                    "Try --boot_re_simple or fewer random slopes."
                )

    if successes == 0:
        raise RuntimeError("Bootstrap failed: no successful refits.")

    slope_mat = np.vstack(slope_draws)
    curv_mat = np.vstack(curv_draws)

    ci_df = pd.DataFrame({
        "target": targets,
        "slope_lo": np.quantile(slope_mat, 0.025, axis=0),
        "slope_hi": np.quantile(slope_mat, 0.975, axis=0),
        "curv_lo": np.quantile(curv_mat, 0.025, axis=0),
        "curv_hi": np.quantile(curv_mat, 0.975, axis=0),
        "n_boot": successes,
        "n_fail": failures,
        "nsim_requested": requested,
    })

    return None, ci_df, {
        "nsim_requested": requested,
        "nsim_success": successes,
        "nsim_failed": failures,
        "seed": seed,
        "fit_method": fit_method,
        "stopped_early": successes < requested,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mixed-effects model of IF proportion (long-run).")
    parser.add_argument("--input_csv", type=str,
                        default="output_GoogleBooks/identity_first_proportion_by_target.csv")
    parser.add_argument("--out_dir", type=str, default="output_GoogleBooks/mixed_effects_long_run")
    parser.add_argument("--min_year", type=int, default=1980)
    parser.add_argument("--max_year", type=int, default=2019)

    # Speed / stability controls
    parser.add_argument("--maxiter", type=int, default=250, help="Max iterations for MixedLM fits.")
    parser.add_argument("--re_simple_main", action="store_true",
                        help="Simplify MAIN random effects to (1 + Year_z | target). Faster + more stable.")

    # Bootstrap controls
    parser.add_argument("--do_bootstrap", action="store_true",
                        help="Run parametric bootstrap for per-target slope/curvature CIs.")
    parser.add_argument("--boot_n", type=int, default=300, help="Max bootstrap iterations attempted.")
    parser.add_argument("--boot_target_success", type=int, default=200,
                        help="Stop once this many successful refits are obtained (faster).")
    parser.add_argument("--boot_seed", type=int, default=123)
    parser.add_argument("--boot_re_simple", action="store_true",
                        help="Simpler RE structure during bootstrap refits: (1 + Year_z | target).")

    args = parser.parse_args()

    # Make it quiet: suppress the usual MixedLM convergence spam
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    commit = get_git_commit_hash()

    # ---------------- Load & prep ----------------
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df):,} rows from {args.input_csv}")

    required_cols = {"target", "Year", "number", "IF_proportion"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    df = df[(df["Year"] >= args.min_year) & (df["Year"] <= args.max_year)].copy()
    if df.empty:
        raise ValueError(f"No data in year range {args.min_year}–{args.max_year}")

    df["Year"] = df["Year"].astype(int)
    df = ensure_number_categorical(df)

    df = df.dropna(subset=["IF_proportion"]).copy()
    df["Year_z"] = zscore(df["Year"])
    df["Year_z2"] = df["Year_z"] ** 2

    year_mean = float(df["Year"].mean())
    year_sd = float(df["Year"].std(ddof=0))

    print(f"After filtering & NA-removal: {len(df):,} rows remain.")
    print(f"Targets: {df['target'].nunique():,}")

    # ---------------- Fit main model (REML) ----------------
    formula = "IF_proportion ~ Year_z + Year_z2 + number"

    if args.re_simple_main:
        re_formula = "~ Year_z"   # (1 + Year_z | target)
        print("\nFitting MAIN model with simplified RE: (1 + Year_z | target)")
    else:
        re_formula = "~ Year_z + Year_z2"
        print("\nFitting MAIN model: (1 + Year_z + Year_z2 | target)")

    model = smf.mixedlm(formula, data=df, groups=df["target"], re_formula=re_formula)

    # Quieter + bounded iterations
    result = model.fit(method="lbfgs", disp=False, maxiter=args.maxiter)
    print(result.summary())

    # Save summary
    summary_path = outdir / "mixed_model_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Git commit: {commit}\n")
        f.write(result.summary().as_text())
        f.write("\n")
    print(f"\nSaved model summary -> {summary_path}")

    # ---------------- LRT for random intercept (ML) ----------------
    # Use start_params to speed refits substantially.
    print("\nRobustness check (ML LRT): random intercept vs no random intercept")
    lrt_path = outdir / "random_intercept_lrt.csv"
    lrt_txt_path = outdir / "random_intercept_lrt.txt"

    lrt_rows = []
    try:
        model_full = smf.mixedlm(formula, data=df, groups=df["target"], re_formula=re_formula)
        res_full_ml = model_full.fit(method="lbfgs", reml=False, disp=False,
                                     maxiter=min(args.maxiter, 200),
                                     start_params=result.params)

        # Reduced: drop random intercept only if re_formula includes intercept.
        # If user chose ~Year_z (intercept+Year_z), reduced becomes 0 + Year_z.
        if re_formula.strip() == "~ Year_z":
            re_formula_red = "0 + Year_z"
        else:
            re_formula_red = "0 + Year_z + Year_z2"

        model_red = smf.mixedlm(formula, data=df, groups=df["target"], re_formula=re_formula_red)
        res_red_ml = model_red.fit(method="lbfgs", reml=False, disp=False,
                                   maxiter=min(args.maxiter, 200),
                                   start_params=res_full_ml.params)

        ll_full = float(res_full_ml.llf)
        ll_red = float(res_red_ml.llf)
        lr_stat = 2.0 * (ll_full - ll_red)
        p_value = float(stats.chi2.sf(lr_stat, df=1))

        msg = (
            f"Git commit: {commit}\n"
            f"ML LRT (approx): LR={lr_stat:.3f}, df=1, p={p_value:.4g}\n"
            f"  ll_full={ll_full:.3f}\n"
            f"  ll_reduced={ll_red:.3f}\n"
            "Note: chi-square reference is approximate (variance on boundary)."
        )
        print(msg)

        lrt_rows.append({
            "model_full": f"(re_formula={re_formula})",
            "model_reduced": f"(re_formula={re_formula_red})",
            "ll_full": ll_full,
            "ll_reduced": ll_red,
            "lr_stat": lr_stat,
            "df": 1,
            "p_value_chi2_approx": p_value,
            "git_commit": commit,
        })

        with open(lrt_txt_path, "w") as f:
            f.write(msg + "\n")

    except Exception as e:
        err = f"Random-intercept LRT failed: {repr(e)}"
        print(err)
        lrt_rows.append({
            "model_full": f"(re_formula={re_formula})",
            "model_reduced": "(random intercept dropped)",
            "ll_full": np.nan,
            "ll_reduced": np.nan,
            "lr_stat": np.nan,
            "df": 1,
            "p_value_chi2_approx": np.nan,
            "error": err,
            "git_commit": commit,
        })
        with open(lrt_txt_path, "w") as f:
            f.write(f"Git commit: {commit}\n{err}\n")

    pd.DataFrame(lrt_rows).to_csv(lrt_path, index=False)
    print(f"Saved LRT table -> {lrt_path}")
    print(f"Saved LRT text  -> {lrt_txt_path}")

    # ---------------- Fixed effects ----------------
    fe = result.params
    fe_se = result.bse
    fe_ci = result.conf_int()

    fe_df = pd.DataFrame({
        "term": fe.index,
        "estimate": fe.values,
        "se": fe_se.values,
        "ci_low": fe_ci[0].values,
        "ci_high": fe_ci[1].values,
        "git_commit": commit,
    })
    fe_path = outdir / "fixed_effects.csv"
    fe_df.to_csv(fe_path, index=False)
    print(f"Saved fixed effects -> {fe_path}")

    # ---------------- Random effects per target ----------------
    re_rows = []
    for target, re_params in result.random_effects.items():
        re_rows.append({
            "target": target,
            "re_Intercept": re_params.get("Intercept", 0.0),
            "re_Year_z": re_params.get("Year_z", 0.0),
            "re_Year_z2": re_params.get("Year_z2", 0.0),
            "git_commit": commit,
        })
    re_df = pd.DataFrame(re_rows).sort_values("target")
    re_path = outdir / "random_effects_by_target.csv"
    re_df.to_csv(re_path, index=False)
    print(f"Saved random effects by target -> {re_path}")

    # ---------------- Derived per-target effects ----------------
    beta0 = float(fe.get("Intercept", np.nan))
    beta1 = float(fe.get("Year_z", np.nan))
    beta2 = float(fe.get("Year_z2", np.nan))
    beta3 = float(fe.get("number[T.singular]", 0.0))

    derived_rows = []
    for _, row in re_df.iterrows():
        t = row["target"]
        b0 = float(row["re_Intercept"])
        b1 = float(row["re_Year_z"])
        b2 = float(row["re_Year_z2"])

        slope = beta1 + b1
        curv = beta2 + b2

        derived_rows.append({
            "target": t,
            "intercept_plural": beta0 + b0,
            "intercept_singular": beta0 + beta3 + b0,
            "slope_plural": slope,
            "slope_singular": slope,
            "curvature_plural": curv,
            "curvature_singular": curv,
            "git_commit": commit,
        })

    derived_df = pd.DataFrame(derived_rows).sort_values("target")
    derived_path = outdir / "derived_effects_by_target.csv"
    derived_df.to_csv(derived_path, index=False)
    print(f"Saved derived per-target effects -> {derived_path}")

    # ---------------- OPTIONAL: Bootstrap CIs (fast) ----------------
    if args.do_bootstrap:
        print("\nRunning FAST parametric bootstrap for per-target slopes/curvature...")

        boot_re_formula = "~ Year_z" if args.boot_re_simple else re_formula

        base_model = smf.mixedlm(
            formula,
            data=df,
            groups=df["target"],
            re_formula=boot_re_formula,
        )

        boot_df, ci_df, meta = parametric_bootstrap_slopes_fast(
            fitted_result=result,
            base_model=base_model,
            df=df,
            group_col="target",
            nsim=int(args.boot_n),
            seed=int(args.boot_seed),
            fit_method="lbfgs",
            target_success=int(args.boot_target_success),
            maxiter=min(args.maxiter, 150),
        )

        ci_path = outdir / f"bootstrap_slopes_ci_success{args.boot_target_success}_max{args.boot_n}.csv"
        meta_path = outdir / f"bootstrap_meta_success{args.boot_target_success}_max{args.boot_n}.txt"

        ci_df.assign(git_commit=commit).to_csv(ci_path, index=False)
        with open(meta_path, "w") as f:
            f.write(f"Git commit: {commit}\n")
            for k, v in meta.items():
                f.write(f"{k}: {v}\n")
            f.write(f"boot_re_formula: {boot_re_formula}\n")

        print(f"Saved bootstrap CIs -> {ci_path}")
        print(f"Saved bootstrap meta -> {meta_path}")

    # ---------------- Predicted trajectories (fixed effects only) ----------------
    cov_fe = np.asarray(result.cov_params())
    fe_index = list(fe.index)
    fe_vec = fe.to_numpy()

    # Identify columns
    idx_intercept = fe_index.index("Intercept") if "Intercept" in fe_index else None
    idx_yearz = fe_index.index("Year_z") if "Year_z" in fe_index else None
    idx_yearz2 = fe_index.index("Year_z2") if "Year_z2" in fe_index else None
    idx_sing = fe_index.index("number[T.singular]") if "number[T.singular]" in fe_index else None

    years = np.arange(args.min_year, args.max_year + 1)
    year_z = (years - year_mean) / year_sd
    year_z2 = year_z ** 2

    rows = []
    x = np.zeros(len(fe_index), dtype=float)

    for y, z, z2 in zip(years, year_z, year_z2):
        x[:] = 0.0
        if idx_intercept is not None:
            x[idx_intercept] = 1.0
        if idx_yearz is not None:
            x[idx_yearz] = float(z)
        if idx_yearz2 is not None:
            x[idx_yearz2] = float(z2)

        # plural
        p_pl, se_pl, lo_pl, hi_pl = fast_predict_ci(fe_vec, cov_fe, x)

        # singular = plural + indicator
        if idx_sing is not None:
            x[idx_sing] = 1.0
        p_sg, se_sg, lo_sg, hi_sg = fast_predict_ci(fe_vec, cov_fe, x)
        if idx_sing is not None:
            x[idx_sing] = 0.0

        rows.append({
            "Year": int(y),
            "pred_plural": p_pl,
            "se_plural": se_pl,
            "ci_low_plural": lo_pl,
            "ci_high_plural": hi_pl,
            "pred_singular": p_sg,
            "se_singular": se_sg,
            "ci_low_singular": lo_sg,
            "ci_high_singular": hi_sg,
            "git_commit": commit,
        })

    traj_df = pd.DataFrame(rows)
    traj_path = outdir / "predicted_trajectories_with_ci.csv"
    traj_df.to_csv(traj_path, index=False)
    print(f"Saved predicted trajectories with 95% CI -> {traj_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
