"""Fase 3: fold rolling origin (P1).

Setiap fold punya cutoff c. Training hanya boleh memakai hari d <= c. Fold memprediksi
horizon c+1 .. c+H. Early stopping memakai `early_stopping_days` terakhir di dalam
jendela train (c-N+1 .. c), tidak pernah horizon.

Peran fold:
  "cv"       fold utama; metriknya dirata-rata dan sebarannya dilaporkan.
  "special"  fold tambahan untuk satu pertanyaan tertentu (ablation Easter, akhir
             tahun); dilaporkan terpisah, tidak pernah dirata-rata ke skor utama.
  "holdout"  tes akhir, dipakai sekali di akhir.

Cara pakai:
    python -m retailfc.folds --config config.yaml     # cetak dan simpan tabel fold
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from retailfc.io import load_calendar, load_config


@dataclass(frozen=True)
class Fold:
    name: str
    role: str            # "cv" | "special" | "holdout"
    cutoff: int          # hari terakhir yang boleh dipakai untuk training dan fitur
    horizon: int
    es_days: int

    @property
    def horizon_start(self) -> int:
        return self.cutoff + 1

    @property
    def horizon_end(self) -> int:
        return self.cutoff + self.horizon

    @property
    def es_start(self) -> int:
        """Hari pertama jendela early stopping (di dalam jendela train)."""
        return self.cutoff - self.es_days + 1

    def fit_mask(self, d: np.ndarray, closed: np.ndarray | None = None) -> np.ndarray:
        """Baris untuk fitting model: sebelum jendela early stopping, bukan hari tutup."""
        m = d < self.es_start
        return m if closed is None else m & ~closed

    def es_mask(self, d: np.ndarray, closed: np.ndarray | None = None) -> np.ndarray:
        """Baris untuk early stopping: es_days terakhir sampai dan termasuk cutoff."""
        m = (d >= self.es_start) & (d <= self.cutoff)
        return m if closed is None else m & ~closed

    def horizon_mask(self, d: np.ndarray) -> np.ndarray:
        """Baris yang diprediksi dan dievaluasi."""
        return (d >= self.horizon_start) & (d <= self.horizon_end)


def build_folds(cfg: dict) -> list[Fold]:
    v = cfg["validation"]
    h, es = v["horizon"], v["early_stopping_days"]
    folds = [Fold(f"cv{i + 1}", "cv", c, h, es) for i, c in enumerate(v["cv_cutoffs"])]
    folds += [Fold(n, "special", c, h, es) for n, c in v.get("extra_folds", {}).items()]
    folds.append(Fold("holdout", "holdout", v["holdout_cutoff"], h, es))
    validate_folds(folds, min_history=v["min_history_days"], last_day=v["holdout_cutoff"] + h)
    return folds


def validate_folds(folds: list[Fold], min_history: int, last_day: int) -> None:
    """Melempar error jika desain melanggar aturan di PROJECT_PLAN.md (Fase 3)."""
    holdout = [f for f in folds if f.role == "holdout"]
    if len(holdout) != 1:
        raise ValueError("exactly one holdout fold is required")
    ho = holdout[0]
    if ho.horizon_end != last_day:
        raise ValueError(f"holdout must end at d_{last_day}, ends at d_{ho.horizon_end}")
    names = [f.name for f in folds]
    if len(set(names)) != len(names):
        raise ValueError("fold names must be unique")

    for f in folds:
        if f.role != "holdout" and f.horizon_end >= ho.horizon_start:
            raise ValueError(f"{f.name}: horizon d_{f.horizon_start}-d_{f.horizon_end} overlaps the holdout")
        if f.es_start - 1 < min_history:
            raise ValueError(f"{f.name}: only {f.es_start - 1} days before early stopping, need {min_history}")

    cv = sorted((f for f in folds if f.role == "cv"), key=lambda f: f.cutoff)
    if len(cv) < 4:
        raise ValueError("at least 4 cv folds are required")
    for a, b in zip(cv, cv[1:]):
        if b.horizon_start <= a.horizon_end:
            raise ValueError(f"cv horizons overlap: {a.name} and {b.name}")


def closed_day_mask(d: np.ndarray, cal: pd.DataFrame, cfg: dict) -> np.ndarray:
    """True untuk hari yang diperlakukan sebagai toko tutup (config `closed_days`)."""
    md = {tuple(x) for x in cfg.get("closed_days", {}).get("month_day", [])}
    c = cal.set_index("d_num")["date"]
    closed_d = {int(d_) for d_, dt in c.items() if (dt.month, dt.day) in md}
    return np.isin(d, list(closed_d))


def fold_table(folds: list[Fold], cal: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    c = cal.set_index("d_num")
    rows = []
    for f in folds:
        h = c.loc[f.horizon_start:f.horizon_end]
        events = sorted({e for col in ("event_name_1", "event_name_2") for e in h[col].dropna()})
        closed = closed_day_mask(np.arange(f.horizon_start, f.horizon_end + 1), cal, cfg)
        rows.append({
            "fold": f.name, "role": f.role,
            "cutoff": f"d_{f.cutoff} ({c.loc[f.cutoff, 'date'].date()})",
            "early_stopping": f"d_{f.es_start}-d_{f.cutoff}",
            "horizon": f"d_{f.horizon_start}-d_{f.horizon_end}",
            "horizon_dates": f"{c.loc[f.horizon_start, 'date'].date()} -> {c.loc[f.horizon_end, 'date'].date()}",
            "closed_days_in_horizon": int(closed.sum()),
            "events_in_horizon": ", ".join(events),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    folds = build_folds(cfg)
    cal = load_calendar(Path(cfg["paths"]["raw_m5"]))
    table = fold_table(folds, cal, cfg)
    print(table.to_string(index=False))
    out = Path(cfg["paths"]["reports"]) / "data" / "folds.csv"
    table.to_csv(out, index=False)
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
