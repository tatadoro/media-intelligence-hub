from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Set

import numpy as np
import pandas as pd

from src.processing.stopwords_build import BuildStopwordsConfig, build_stopwords_ru_v001


# Минимальный “жёсткий” бан-лист функциональных слов/частиц/местоимений.
# Его задача — не дать доменной эвристике тащить в auto-листы базовую грамматику.
_COMMON_RU_FUNCTION_WORDS: Set[str] = {
    "в", "и", "на", "что", "по", "с", "не", "о", "а", "к", "у", "от", "до", "но", "за",
    "из", "под", "при", "без", "для", "об", "же", "ли", "бы",
    "он", "она", "они", "его", "ее", "их", "ему", "ей", "ним", "них",
    "я", "мы", "вы", "ты", "мне", "тебе", "вам", "нас", "вас",
    "это", "как", "так", "уже", "еще", "её", "все", "всё",
    # часто встречающиеся “служебные” в новостях (можно расширять по факту):
    "год", "года", "лет", "вс",
}


@dataclass
class DomainStopCfg:
    # домен: терм часто встречается в документах домена
    df_ratio_hi: float = 0.15
    # и редко встречается в остальных RSS
    df_ratio_lo: float = 0.08
    # минимум частоты в домене
    tf_min: int = 10
    max_terms: int = 200

    # агрегация df_ratio по остальным RSS (устойчивее, чем mean при перекосах)
    df_ratio_lo_agg: Literal["mean", "p95"] = "p95"

    # фильтровать “общие” термины
    filter_common_terms: bool = True
    common_df_ratio_min: float = 0.30
    common_tf_min: int = 50

    # жёсткий бан-лист (можно расширять извне)
    hard_exclude_terms: Set[str] = field(default_factory=lambda: set(_COMMON_RU_FUNCTION_WORDS))


def _agg_df_ratio(series: pd.Series, how: str) -> float:
    s = series.dropna().astype(float)
    if s.empty:
        return 0.0
    if how == "mean":
        return float(s.mean())
    return float(np.quantile(s.to_numpy(), 0.95))


def domain_specific_terms(
    stats: pd.DataFrame,
    domain_source: str,
    cfg: DomainStopCfg,
    lang: str = "unknown",
) -> pd.DataFrame:
    out_cols = ["term", "tf_hi", "df_ratio_hi", "tf_lo", "df_ratio_lo"]

    if stats is None or getattr(stats, "empty", True):
        return pd.DataFrame(columns=out_cols)

    need_cols = {"source", "source_type", "lang", "term", "tf", "df_ratio"}
    missing = sorted(list(need_cols - set(stats.columns)))
    if missing:
        raise ValueError(f"stats missing columns: {missing}")

    s = stats.copy()
    s["source"] = s["source"].astype(str)
    s["source_type"] = s["source_type"].astype(str)
    s["lang"] = s["lang"].fillna("unknown").astype(str)
    s["term"] = s["term"].fillna("").astype(str).str.strip()

    # только RSS и нужный lang
    s = s[(s["source_type"] == "rss") & (s["lang"] == lang)]
    if s.empty:
        return pd.DataFrame(columns=out_cols)

    dom = str(domain_source)

    # HI: конкретный домен
    hi = s[s["source"] == dom][["term", "tf", "df_ratio"]].copy()
    if hi.empty:
        return pd.DataFrame(columns=out_cols)

    hi.rename(columns={"tf": "tf_hi", "df_ratio": "df_ratio_hi"}, inplace=True)

    # LO: остальные RSS
    lo_raw = s[s["source"] != dom][["term", "tf", "df_ratio"]].copy()

    if lo_raw.empty:
        lo = pd.DataFrame({"term": hi["term"].unique(), "tf_lo": 0, "df_ratio_lo": 0.0})
    else:
        tf_lo = lo_raw.groupby("term", as_index=False)["tf"].sum().rename(columns={"tf": "tf_lo"})

        df_ratio_lo = (
            lo_raw.groupby("term", as_index=False)["df_ratio"]
            .apply(lambda x: _agg_df_ratio(x, cfg.df_ratio_lo_agg))
            .rename(columns={"df_ratio": "df_ratio_lo"})
        )

        lo = tf_lo.merge(df_ratio_lo, on="term", how="outer").fillna({"tf_lo": 0, "df_ratio_lo": 0.0})

    out = hi.merge(lo, on="term", how="left").fillna({"tf_lo": 0, "df_ratio_lo": 0.0})

    # 0) жёстко исключаем базовые функциональные слова
    if cfg.hard_exclude_terms:
        out = out[~out["term"].str.lower().isin({t.lower() for t in cfg.hard_exclude_terms})].copy()
        if out.empty:
            return pd.DataFrame(columns=out_cols)

    # 1) базовые условия доменности
    out = out[
        (out["df_ratio_hi"] >= cfg.df_ratio_hi)
        & (out["df_ratio_lo"] <= cfg.df_ratio_lo)
        & (out["tf_hi"] >= cfg.tf_min)
    ].copy()

    if out.empty:
        return pd.DataFrame(columns=out_cols)

    # 2) фильтр “общих” по остальным RSS + по базовому стоплисту (v001, без доменных авто!)
    if cfg.filter_common_terms:
        base_sw = set(build_stopwords_ru_v001(BuildStopwordsConfig()))

        common_by_lo = set(
            lo.loc[
                (lo["df_ratio_lo"] >= float(cfg.common_df_ratio_min))
                & (lo["tf_lo"] >= int(cfg.common_tf_min)),
                "term",
            ].astype(str).str.lower()
        )

        common = {t.lower() for t in base_sw} | common_by_lo
        out = out[~out["term"].str.lower().isin(common)].copy()

    if out.empty:
        return pd.DataFrame(columns=out_cols)

    out.sort_values(["df_ratio_hi", "tf_hi"], ascending=[False, False], inplace=True)
    return out[out_cols].head(int(cfg.max_terms))