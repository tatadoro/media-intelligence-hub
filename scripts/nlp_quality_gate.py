from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ratio_ok(ok_key: str, ratio_key: str, block: dict, min_ratio: float, errors: list[str]) -> None:
    total_key = "total"
    ratio = float(block.get(ratio_key, 0.0))
    total = int(block.get(total_key, 0))
    ok = int(block.get(ok_key, 0))
    if total == 0:
        errors.append(f"[FAIL] {ratio_key}: empty dataset")
        return
    if ratio < min_ratio:
        errors.append(f"[FAIL] {ratio_key}: {ok}/{total} ({ratio:.0%}) < {min_ratio:.0%}")


def main() -> int:
    report_path = Path(os.getenv("NLP_REPORT_JSON", "reports/nlp_coverage_report_2026-02-03.json"))
    if not report_path.exists():
        print(f"[ERROR] Report not found: {report_path}")
        return 2

    data = _load(report_path)
    thresholds = data.get("thresholds", {})
    lang_min = float(thresholds.get("lang_ok_ratio_min", os.getenv("NLP_LANG_OK_MIN", "0.9")))
    kp_min = float(thresholds.get("keyphrases_ok_ratio_min", os.getenv("NLP_KP_OK_MIN", "0.8")))
    sent_min = float(thresholds.get("sentiment_ok_ratio_min", os.getenv("NLP_SENT_OK_MIN", "0.15")))

    errors: list[str] = []
    for section in ("rss", "telegram"):
        block = data.get(section, {})
        _ratio_ok("lang_ok", "lang_ok_ratio", block, lang_min, errors)
        _ratio_ok("keyphrases_ok", "keyphrases_ok_ratio", block, kp_min, errors)
        _ratio_ok("sentiment_ok", "sentiment_ok_ratio", block, sent_min, errors)

    if errors:
        print("\n".join(errors))
        return 1

    print("[OK] NLP quality gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
