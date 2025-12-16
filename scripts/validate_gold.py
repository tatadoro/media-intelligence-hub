from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


REQUIRED_COLS = (
    "id",
    "title",
    "link",
    "published_at",
    "source",
    "clean_text",
    "summary",
    "keywords",
)

OPTIONAL_TEXT_COLS = ("nlp_text", "raw_text")


def main() -> int:
    p = argparse.ArgumentParser(description="Validate gold Parquet contract.")
    p.add_argument("--input", required=True, help="Path to gold *_processed.parquet")
    p.add_argument("--min-rows", type=int, default=1, help="Minimum number of rows")
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

    if len(df) < args.min_rows:
        print(f"[ERROR] Gold contract failed: rows={len(df)} < {args.min_rows} ({path})", file=sys.stderr)
        return 1

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"[ERROR] Gold contract failed: missing cols {missing} ({path})", file=sys.stderr)
        return 1

    # мягкие проверки (не валим пайплайн)
    empty_summary = (df["summary"].astype(str).str.len() == 0).mean()
    empty_keywords = (df["keywords"].astype(str).str.len() == 0).mean()

    has_any_text = False
    for c in ("clean_text",) + OPTIONAL_TEXT_COLS:
        if c in df.columns:
            share_nonempty = (df[c].astype(str).str.len() > 0).mean()
            if share_nonempty > 0:
                has_any_text = True
            break

    if not has_any_text:
        print("[WARN] No non-empty text columns detected (clean_text/nlp_text/raw_text).")

    print(f"[OK] Gold contract valid: {path}")
    print(f"[INFO] rows={len(df)} empty_summary_share={empty_summary:.3f} empty_keywords_share={empty_keywords:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())