from __future__ import annotations
from dataclasses import dataclass
import re
import pandas as pd


@dataclass
class CandidateConfig:
    df_ratio_min: float = 0.60
    tf_min: int = 50
    max_terms: int = 2000


_BAD_RE = re.compile(r"\d")  # с цифрами пока выкидываем


def build_candidates(stats: pd.DataFrame, cfg: CandidateConfig) -> pd.DataFrame:
    s = stats.copy()
    s["term"] = s["term"].astype(str)
    s = s[~s["term"].str.contains(_BAD_RE, na=False)]
    s = s[(s["df_ratio"] >= cfg.df_ratio_min) & (s["tf"] >= cfg.tf_min)]
    s = s.sort_values(["df_ratio", "tf"], ascending=[False, False]).head(cfg.max_terms)
    return s.reset_index(drop=True)