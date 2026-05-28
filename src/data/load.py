"""Dataset loading. Single switch: `source` = 'synthetic' | 'mimic4'."""
from __future__ import annotations

import pandas as pd

from src.config import PROCESSED_DIR, SYNTHETIC_DIR


def load_icu_stays(source: str = "synthetic", *, refresh: bool = False) -> pd.DataFrame:
    if source == "synthetic":
        path = SYNTHETIC_DIR / "icu_stays_synth.parquet"
        if not path.exists() or refresh:
            from src.data.synthetic import main as gen
            gen()
        return pd.read_parquet(path)

    if source == "mimic4":
        cache = PROCESSED_DIR / "mimic4_full_icu_stays.parquet"
        if cache.exists() and not refresh:
            return pd.read_parquet(cache)
        from src.data.mimic import build_dataset
        df = build_dataset()
        df.to_parquet(cache, index=False)
        return df

    raise NotImplementedError(f"source={source!r} not wired yet")
