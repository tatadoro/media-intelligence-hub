from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def share_nonempty(series: pd.Series, min_chars: int = 1) -> float:
    s = series.fillna("").astype(str)
    return float((s.str.len() >= min_chars).mean())


def main() -> int:
    p = argparse.ArgumentParser(description="Quality gate for gold parquet (best-effort by default).")
    p.add_argument("--input", required=True, help="Path to gold *_processed.parquet")
    p.add_argument("--min-rows", type=int, default=20)
    p.add_argument("--min-body-share", type=float, default=0.05, help="Share of rows with non-empty clean_text")
    p.add_argument("--min-body-chars", type=int, default=50, help="clean_text length threshold for 'has body'")
    p.add_argument("--max-empty-summary-share", type=float, default=0.95)
    p.add_argument("--max-empty-keywords-share", type=float, default=0.95)
    p.add_argument("--strict", action="store_true", help="Fail (exit 1) if gate conditions fail")
    args = p.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        return 2

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"[ERROR] Cannot read parquet: {path}\n{e}", file=sys.stderr)
        return 2

    rows = len(df)
    if rows == 0:
        msg = f"[ERROR] Gate: rows=0 ({path})"
        print(msg, file=sys.stderr)
        return 1 if args.strict else 0

    # Метрики (если колонки отсутствуют — считаем как "плохо", но не падаем чтением)
    clean_text = df["clean_text"] if "clean_text" in df.columns else pd.Series([""] * rows)
    summary = df["summary"] if "summary" in df.columns else pd.Series([""] * rows)
    keywords = df["keywords"] if "keywords" in df.columns else pd.Series([""] * rows)

    body_share = share_nonempty(clean_text, min_chars=args.min_body_chars)
    empty_summary_share = 1.0 - share_nonempty(summary, min_chars=1)
    empty_keywords_share = 1.0 - share_nonempty(keywords, min_chars=1)

    print(f"[INFO] gate file={path}")
    print(f"[INFO] rows={rows} body_share={body_share:.3f} empty_summary_share={empty_summary_share:.3f} empty_keywords_share={empty_keywords_share:.3f}")

    failures: list[str] = []

    if rows < args.min_rows:
        failures.append(f"rows {rows} < min_rows {args.min_rows}")
    if body_share < args.min_body_share:
        failures.append(f"body_share {body_share:.3f} < min_body_share {args.min_body_share:.3f} (min_body_chars={args.min_body_chars})")
    if empty_summary_share > args.max_empty_summary_share:
        failures.append(f"empty_summary_share {empty_summary_share:.3f} > max {args.max_empty_summary_share:.3f}")
    if empty_keywords_share > args.max_empty_keywords_share:
        failures.append(f"empty_keywords_share {empty_keywords_share:.3f} > max {args.max_empty_keywords_share:.3f}")

    if failures:
        print("[WARN] Quality gate FAILED conditions:")
        for f in failures:
            print(f"  - {f}")
        if args.strict:
            print("[ERROR] STRICT=1: failing pipeline.", file=sys.stderr)
            return 1
        print("[WARN] STRICT=0: continuing (best-effort).")
        return 0

    print("[OK] Quality gate PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())