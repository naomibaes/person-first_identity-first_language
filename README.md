# Identity-first vs person-first language 

This repository contains a complete, reproducible pipeline for analysing long-run trends in **identity-first (IF)** versus **person-first (PF)** language in printed English using **Google Books Ngrams** (1940–2019).

The pipeline extracts normalized term frequencies, constructs identity-first usage indices, visualizes temporal trends, and estimates long-run change using mixed-effects models.

---

## What this repository does

At a high level, the workflow:

1. Queries Google Books Ngrams for identity-first and person-first term variants  
2. Computes identity-first (IF) proportions by year and grammatical number  
3. Aggregates IF usage using both unweighted and frequency-weighted strategies  
4. Models long-run trends using linear mixed-effects regression  

All frequencies are **year-normalized relative frequencies** returned directly by the Google Books Ngrams JSON endpoint (no API key required).

---

## Pipeline overview

You can run scripts from the notebook: "GoogleBooks_ngrams.ipynb"

### 1. Data extraction

**Script:** `0_google_ngrams_ntf.py`  
**Input:** `input/targets.csv`  

**Outputs:**
- `output_GoogleBooks/ngram_relative_wide.csv`
- `output_GoogleBooks/ngram_relative_long.csv`

This step queries the (unofficial but stable) Google Books Ngrams JSON endpoint and retrieves yearly normalized frequencies (`ntf`) for each identity-first and person-first term variant.

---

### 2. Computing the IF index

**Script:** `2_compute_index_GB.py`  
**Input:** `output_GoogleBooks/ngram_relative_long.csv`  

**Outputs:**
- `identity_first_proportion_by_target.csv`
- `identity_first_proportion_overall_unweighted.csv`
- `identity_first_proportion_overall_weighted.csv`

The primary dependent variable is the **identity-first (IF) proportion**:

\[
\text{IF proportion} = \frac{\text{IF}}{\text{IF} + \text{PF}}
\]

This index is computed separately for each **year** and **grammatical number** (singular vs plural).

Two aggregation strategies are provided:
- **Unweighted:** mean of per-target IF proportions (each diagnostic label contributes equally)
- **Weighted:** IF and PF frequencies are pooled across targets before computing the proportion (targets weighted by frequency of mention)

---

### 3. Plotting

**Scripts:**
- `1_plot_ntf_GB.py` — raw term frequencies
- `2.5_plot_index_GB.py` — per-target IF trajectories
- `2.5_plot_index_overall_GB.py` — overall IF trajectories (weighted vs unweighted)

**Outputs:**  
Saved to `output_GoogleBooks/plots/`

Figures include:
- Per-target IF trends (singular vs plural)
- Overall IF proportion over time
- Weighted vs unweighted comparisons

---

### 4. Statistical modelling

**Script:** `3_statistics_GB.py`  
**Input:** `identity_first_proportion_by_target.csv`  

**Outputs:**
- `mixed_effects_long_run/derived_effects_by_target.csv`
- `mixed_effects_long_run/predicted_trajectories_with_ci.csv`

A single **linear mixed-effects model** is fit across all targets:

- **DV:** IF proportion  
- **Fixed effects:** linear and quadratic year trends, grammatical number  
- **Random effects:** target-level intercepts and slopes  

This allows evaluation of:
- Overall long-run change in IF usage
- Singular vs plural differences
- Condition-level baseline differences
- Condition-level differences in temporal trajectories

---

## Notes

- Frequencies are proportions of all tokens per year (values are typically ~1e-6).
- Results depend on corpus choice (default: English 2019, `corpus_id = 26`).
- The Google Books Ngrams endpoint is unofficial; results are cached locally for reproducibility.
