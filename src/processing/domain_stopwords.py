from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class DomainStopCfg:
    df_ratio_hi: float = 0.20   # терм "почти в каждом" документе домена
    df_ratio_lo: float = 0.03   # и почти отсутствует в остальных rss
    tf_min: int = 20            # чтобы отсечь случайные редкие
    max_terms: int = 200        # верхний лимит


def domain_specific_terms(
    stats: pd.DataFrame,
    domain_source: str,
    cfg: DomainStopCfg,
    lang: str = "unknown",
) -> pd.DataFrame:
    """
    stats: DataFrame из stats_by_source.parquet
      cols: source, source_type, lang, term, tf, df, df_ratio

    domain_source: например "rss:tvzvezda.ru"
    Возвращает: term, tf_hi, df_ratio_hi, tf_lo, df_ratio_lo
    """
    if stats.empty:
        return pd.DataFrame(columns=["term", "tf_hi", "df_ratio_hi", "tf_lo", "df_ratio_lo"])

    need_cols = {"source", "source_type", "lang", "term", "tf", "df_ratio"}
    missing = sorted(list(need_cols - set(stats.columns)))
    if missing:
        raise ValueError(f"stats missing columns: {missing}")

    s = stats.copy()
    s["source"] = s["source"].astype(str)
    s["source_type"] = s["source_type"].astype(str)
    s["lang"] = s["lang"].fillna("unknown").astype(str)

    # Берём только rss + нужный lang
    s = s[(s["source_type"] == "rss") & (s["lang"] == lang)]
    if s.empty:
        return pd.DataFrame(columns=["term", "tf_hi", "df_ratio_hi", "tf_lo", "df_ratio_lo"])

    hi = s[s["source"] == domain_source][["term", "tf", "df_ratio"]].copy()
    if hi.empty:
        return pd.DataFrame(columns=["term", "tf_hi", "df_ratio_hi", "tf_lo", "df_ratio_lo"])

    hi.rename(columns={"tf": "tf_hi", "df_ratio": "df_ratio_hi"}, inplace=True)

    lo = (
        s[s["source"] != domain_source]
        .groupby("term", as_index=False)
        .agg(tf_lo=("tf", "sum"), df_ratio_lo=("df_ratio", "mean"))
    )

    out = hi.merge(lo, on="term", how="left").fillna({"tf_lo": 0, "df_ratio_lo": 0.0})

    out = out[
        (out["df_ratio_hi"] >= cfg.df_ratio_hi)
        & (out["df_ratio_lo"] <= cfg.df_ratio_lo)
        & (out["tf_hi"] >= cfg.tf_min)
    ].copy()

    out.sort_values(["df_ratio_hi", "tf_hi"], ascending=[False, False], inplace=True)
    return out.head(cfg.max_terms)