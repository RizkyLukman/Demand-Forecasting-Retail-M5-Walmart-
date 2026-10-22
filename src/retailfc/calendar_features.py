"""Fase 4: fitur kalender, termasuk fitur hari libur bergerak (P4).

Semua fitur di sini hanya membaca kalender (tanggal, event, SNAP), yang sudah diketahui
jauh hari. Fungsi tetap menerima `cutoff` agar seragam dengan fitur lain dan bisa diuji
dengan `leakage.leaking_columns`.

`days_to_event` / `days_since_event` dibuat generik: dipakai untuk Easter di M5, dan
nanti untuk Idulfitri di data Indonesia (Fase 8).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EVENT_COLS = ["event_name_1", "event_name_2"]


def event_days(cal: pd.DataFrame, name: str) -> np.ndarray:
    """Nomor hari (d) tempat `name` muncul di `event_name_1` ATAU `event_name_2`."""
    m = np.zeros(len(cal), dtype=bool)
    for col in EVENT_COLS:
        m |= (cal[col] == name).to_numpy()
    return np.sort(cal.loc[m, "d_num"].to_numpy().astype(np.int32))


def days_to_event(d: np.ndarray, event_d: np.ndarray, clip: int) -> np.ndarray:
    """Hari menuju event berikutnya (0 = hari event), dipotong di `clip`.

    Nilai `clip` berarti "jauh dari event" (>= clip hari, atau tidak ada event berikutnya).
    """
    d = np.asarray(d, dtype=np.int32)
    idx = np.searchsorted(event_d, d, side="left")          # event pertama >= d
    nxt = np.where(idx < len(event_d), event_d[np.minimum(idx, len(event_d) - 1)], np.iinfo(np.int32).max)
    return np.minimum(nxt.astype(np.int64) - d, clip).astype(np.int8)


def days_since_event(d: np.ndarray, event_d: np.ndarray, clip: int) -> np.ndarray:
    """Hari sejak event terakhir (0 = hari event), dipotong di `clip`."""
    d = np.asarray(d, dtype=np.int32)
    idx = np.searchsorted(event_d, d, side="right") - 1     # event terakhir <= d
    prv = np.where(idx >= 0, event_d[np.maximum(idx, 0)], np.iinfo(np.int32).min // 2)
    return np.minimum(d - prv.astype(np.int64), clip).astype(np.int8)


def calendar_table(cal: pd.DataFrame, state: str, moving_events: dict[str, int]) -> pd.DataFrame:
    """Satu baris per hari (index = d_num) berisi semua fitur kalender untuk satu negara bagian.

    `moving_events`: {nama_event: clip}, mis. {"Easter": 7}.
    """
    d = cal["d_num"].to_numpy()
    names = sorted(set(cal["event_name_1"].dropna()) | set(cal["event_name_2"].dropna()))
    types = sorted(set(cal["event_type_1"].dropna()) | set(cal["event_type_2"].dropna()))
    out = pd.DataFrame({
        "wday": cal["wday"].to_numpy().astype(np.int8),          # 1 = Sabtu (konvensi M5)
        "mday": cal["date"].dt.day.to_numpy().astype(np.int8),
        "month": cal["month"].to_numpy().astype(np.int8),
        "year": cal["year"].to_numpy().astype(np.int16),
        "event_name_1": pd.Categorical(cal["event_name_1"], categories=names),
        "event_type_1": pd.Categorical(cal["event_type_1"], categories=types),
        "event_name_2": pd.Categorical(cal["event_name_2"], categories=names),
        "snap": cal[f"snap_{state}"].to_numpy().astype(np.int8),
    }, index=pd.Index(d, name="d"))
    for name, clip in moving_events.items():
        ev = event_days(cal, name)
        key = name.lower()
        out[f"days_to_{key}"] = days_to_event(d, ev, clip)
        out[f"days_since_{key}"] = days_since_event(d, ev, clip)
    return out


def calendar_features(df: pd.DataFrame, cutoff: int, cal_table: pd.DataFrame) -> pd.DataFrame:
    """Gabungkan `calendar_table` ke baris `df` berdasarkan kolom `d`. `cutoff` tidak dipakai:
    kalender sudah diketahui sebelumnya."""
    out = cal_table.reindex(df["d"].to_numpy())
    out.index = df.index
    return out
