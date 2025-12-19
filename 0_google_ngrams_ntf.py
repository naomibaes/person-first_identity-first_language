#!/usr/bin/env python3
# Authors: Naomi Baes & ChatGPT
# Purpose: Pull relative (normalized) frequencies for identity-first & person-first terms
#          from Google Books Ngram JSON API based on a targets.csv schema.

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests

NGRAM_ENDPOINT = "https://books.google.com/ngrams/json"

def load_targets(
    csv_path: str,
    form_cols: Tuple[str, str, str, str] = (
        "identity_first_singular",
        "identity_first_plural",
        "person_first_singular",
        "person_first_plural",
    ),
) -> Tuple[pd.DataFrame, List[Tuple[str, str, str, str, int]]]:
    """
    Read targets.csv and return:
      (df, expanded_rows)
      expanded_rows: list of (target, form_type, term, group_key, control)
        - form_type in {"IF_sg","IF_pl","PF_sg","PF_pl"}
        - group_key is the 'target' string (e.g., "schizophrenic").
      Multiple variants are split by '|' in each form column.
    """
    df = pd.read_csv(csv_path)
    required = {"target", "control", *form_cols}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"targets.csv is missing columns: {missing}")

    mapping = {
        "identity_first_singular": "IF_sg",
        "identity_first_plural": "IF_pl",
        "person_first_singular": "PF_sg",
        "person_first_plural": "PF_pl",
    }

    expanded: List[Tuple[str, str, str, str, int]] = []
    for _, row in df.iterrows():
        group_key = str(row["target"])
        control = int(row["control"])
        for col in form_cols:
            form_type = mapping[col]
            cell = "" if pd.isna(row[col]) else str(row[col])
            # split on '|' with or without spaces
            terms = [t.strip() for t in cell.split("|") if t and t.strip()]
            for term in terms:
                expanded.append((group_key, form_type, term, group_key, control))
    return df, expanded


def chunked(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def fetch_batch(
    terms: List[str],
    year_start: int,
    year_end: int,
    corpus_id: int,
    smoothing: int,
    case_insensitive: bool,
    session: requests.Session,
    timeout: int = 30,
    retries: int = 3,
    backoff: float = 0.8,
) -> Dict[str, List[float]]:
    """
    Fetch a batch of terms in one request.
    Returns dict: term -> list of normalized frequencies for each year.
    Missing terms get zero series of expected length.
    """
    params = {
        "content": ",".join(terms),
        "year_start": year_start,
        "year_end": year_end,
        "corpus": corpus_id,         # e.g., 26 = eng_2019
        "smoothing": smoothing,      # integer
    }
    if case_insensitive:
        params["case_insensitive"] = "on"  # API expects "on" if enabled

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    data = []
    for attempt in range(1, retries + 1):
        try:
            r = session.get(NGRAM_ENDPOINT, params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            if attempt == retries:
                raise RuntimeError(f"Failed after {retries} attempts: {e}")
            time.sleep(backoff * attempt)

    expected_len = year_end - year_start + 1
    out: Dict[str, List[float]] = {t: [0.0] * expected_len for t in terms}

    # API returns a list of objects with keys 'ngram' and 'timeseries'
    # Some docs/examples also show 'ngram_case_insensitive'; prefer that label if present.
    for obj in data:
        label = obj.get("ngram_case_insensitive") or obj.get("ngram") or ""
        ts = obj.get("timeseries", []) or []
        if len(ts) != expected_len:
            ts = (ts + [0.0] * expected_len)[:expected_len]

        # map back to the requested term list conservatively
        if label in out:
            out[label] = ts
        else:
            # fallback: case-insensitive align
            lbl_lower = label.lower()
            for t in terms:
                if t.lower() == lbl_lower:
                    out[t] = ts
                    break
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Get relative frequencies for terms listed in targets.csv from Google Books Ngrams JSON."
    )
    ap.add_argument("--targets_csv", type=str, default="targets.csv",
                    help="Path to targets.csv (must include: target, control, and IF/PF form columns).")
    ap.add_argument("--out_wide_csv", type=str, default="data/output_GoogleBooks/ngram_relative_wide.csv",
                    help="Output wide CSV: Year + one <term>_ntf column per term.")
    ap.add_argument("--out_long_csv", type=str, default="data/output/ngram_relative_long.csv",
                    help="Output long CSV: Year,target,form_type,term,ntf,control.")
    ap.add_argument("--year_start", type=int, default=1940)
    ap.add_argument("--year_end", type=int, default=2019)
    ap.add_argument("--corpus_id", type=int, default=26,
                    help="Google Ngram corpus ID (e.g., 26 = eng_2019, 27 = eng-us_2019, 28 = eng-gb_2019, 29 = eng-fiction_2019).")
    ap.add_argument("--smoothing", type=int, default=3,
                    help="Viewer-side smoothing (integer, e.g., 0–5).")
    ap.add_argument("--case_insensitive", action="store_true",
                    help="If set, merge case variants via API’s case_insensitive=on.")
    ap.add_argument("--batch_size", type=int, default=8,
                    help="Max ~8 terms per request is safest to avoid URL-length issues.")
    ap.add_argument("--sleep", type=float, default=0.25,
                    help="Seconds to sleep between API calls (politeness).")
    args = ap.parse_args()

    # 1) Load and expand targets
    _, expanded = load_targets(args.targets_csv)
    # expanded entries: (group_key, form_type, term, group_key, control)
    terms = [t for (_, _, t, _, _) in expanded]

    # 2) Fetch normalized series per unique term
    unique_terms = list(dict.fromkeys(terms))  # de-dupe preserving order
    years = list(range(args.year_start, args.year_end + 1))
    sess = requests.Session()

    term_to_series: Dict[str, List[float]] = {}
    for batch in chunked(unique_terms, args.batch_size):
        resp = fetch_batch(
            batch,
            year_start=args.year_start,
            year_end=args.year_end,
            corpus_id=args.corpus_id,
            smoothing=args.smoothing,
            case_insensitive=args.case_insensitive,
            session=sess,
        )
        term_to_series.update(resp)
        if args.sleep > 0:
            time.sleep(args.sleep)

    # 3) Build wide table (Year + one column per term as <term>_ntf)
    wide_df = pd.DataFrame({"Year": years})
    for term in unique_terms:
        wide_df[f"{term}_ntf"] = term_to_series.get(term, [0.0] * len(years))

    # 4) Build tidy long table
    long_rows = []
    for group_key, form_type, term, _, control in expanded:
        series = term_to_series.get(term, [0.0] * len(years))
        for offset, ntf in enumerate(series):
            long_rows.append({
                "Year": args.year_start + offset,
                "target": group_key,
                "form_type": form_type,   # IF_sg / IF_pl / PF_sg / PF_pl
                "term": term,
                "ntf": ntf,
                "control": control
            })
    long_df = pd.DataFrame(long_rows, columns=["Year", "target", "form_type", "term", "ntf", "control"])

    # 5) Save
    Path(args.out_wide_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_long_csv).parent.mkdir(parents=True, exist_ok=True)
    wide_df.to_csv(args.out_wide_csv, index=False)
    long_df.to_csv(args.out_long_csv, index=False)

    print(f"Wrote WIDE relative frequencies -> {args.out_wide_csv}")
    print(f"Wrote LONG relative frequencies -> {args.out_long_csv}")


if __name__ == "__main__":
    main()
