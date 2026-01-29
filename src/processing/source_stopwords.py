from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class SourceStopCfg:
    df_ratio_hi: float = 0.25
    df_ratio_lo: float = 0.03
    tf_min: int = 30
    max_terms: int = 500


def source_specific_terms(
    stats_by_source: pd.DataFrame,
    src_hi: str,
    src_lo: str,
    cfg: SourceStopCfg,
    lang: str = "unknown",
) -> pd.DataFrame:
    """
    Ищем термины, характерные для src_hi и нехарактерные для src_lo.
    Работает на stats_by_source из mine_tf_df_by_source.

    На старте используем lang='unknown', потому что у тебя в silver сейчас весь язык unknown.
    Когда починим lang, просто вызовем для lang='ru'.
    """
    s = stats_by_source.copy()
    s["source_type"] = s["source_type"].astype(str)
    s["lang"] = s["lang"].fillna("unknown").astype(str)

    hi = s[(s["source_type"] == src_hi) & (s["lang"] == lang)][["term", "tf", "df_ratio"]].rename(
        columns={"tf": "tf_hi", "df_ratio": "df_ratio_hi"}
    )
    lo = s[(s["source_type"] == src_lo) & (s["lang"] == lang)][["term", "tf", "df_ratio"]].rename(
        columns={"tf": "tf_lo", "df_ratio": "df_ratio_lo"}
    )

    m = hi.merge(lo, on="term", how="left").fillna({"tf_lo": 0, "df_ratio_lo": 0.0})

    m = m[
        (m["df_ratio_hi"] >= cfg.df_ratio_hi)
        & (m["df_ratio_lo"] <= cfg.df_ratio_lo)
        & (m["tf_hi"] >= cfg.tf_min)
    ]

    m = m.sort_values(["df_ratio_hi", "tf_hi"], ascending=[False, False]).head(cfg.max_terms)
    return m.reset_index(drop=True)