"""Fase 8: baca dump SQL data apotek Indonesia (Mendeley 2ym7v78wtd v1) ke Parquet.

Sumber: Gustriansyah, R. (2022). Retail sales dataset of a pharmacy in Indonesia.
Mendeley Data, V1. DOI 10.17632/2ym7v78wtd.1. Lisensi CC BY 4.0.

File mentah: data/raw/id_pharmacy/sales.zip -> sales.sql (dump MariaDB dari HeidiSQL).
Tiap baris data berbentuk satu tuple per baris teks, misalnya
    ('RJ-01.2015-08-0001', 'AI-1157', 1, 1335, 1615.35, 10),
Parser membaca tuple apa adanya. Jumlah kolom dicocokkan dengan daftar kolom di
pernyataan INSERT; baris yang tidak cocok membuat parser berhenti, bukan dilewati.

Output: data/interim/id_pharmacy/<tabel>.parquet (det_sales, ms_product, ms_sales,
transaction). Nilai disimpan seperti di sumber; pembersihan dilakukan di tahap
berikutnya dan dicatat di laporan audit.

Cara pakai:
    python -m retailfc.id_pharmacy --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from retailfc.io import load_config

INSERT_RE = re.compile(r"^INSERT INTO `(\w+)` \(([^)]*)\) VALUES\s*$")
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
ESCAPES = {"0": "\0", "n": "\n", "r": "\r", "t": "\t", "Z": "\x1a", "b": "\b"}

# Kolom bertipe tanggal/waktu di skema dump.
DATE_COLS = {"ms_sales": ["TGL"], "transaction": ["TGL"]}


def parse_tuple(line: str) -> tuple[list, str]:
    """Parse satu tuple '(v1, v2, ...)' di awal `line`; kembalikan (nilai, sisa teks)."""
    s = line.lstrip()
    if not s.startswith("("):
        raise ValueError(f"tuple harus diawali '(': {line[:80]!r}")
    i, n, vals = 1, len(s), []
    while True:
        while i < n and s[i] == " ":
            i += 1
        if s[i] == "'":
            i += 1
            buf = []
            while True:
                c = s[i]
                if c == "\\":
                    nxt = s[i + 1]
                    buf.append(ESCAPES.get(nxt, nxt))
                    i += 2
                elif c == "'":
                    if i + 1 < n and s[i + 1] == "'":  # '' = tanda kutip di dalam string
                        buf.append("'")
                        i += 2
                    else:
                        i += 1
                        break
                else:
                    buf.append(c)
                    i += 1
            vals.append("".join(buf))
        elif s.startswith("NULL", i):
            vals.append(None)
            i += 4
        else:
            m = NUMBER_RE.match(s, i)
            if not m:
                raise ValueError(f"nilai tidak dikenal di posisi {i}: {s[i:i + 40]!r}")
            txt = m.group(0)
            vals.append(float(txt) if any(ch in txt for ch in ".eE") else int(txt))
            i = m.end()
        while i < n and s[i] == " ":
            i += 1
        if s[i] == ",":
            i += 1
            continue
        if s[i] == ")":
            return vals, s[i + 1:]
        raise ValueError(f"pemisah tidak dikenal di posisi {i}: {s[i:i + 40]!r}")


def parse_dump(path: Path) -> dict[str, pd.DataFrame]:
    """Baca semua tabel dari dump. Gagal keras jika ada baris yang tidak bisa dipahami."""
    rows: dict[str, list[list]] = {}
    cols: dict[str, list[str]] = {}
    table = None
    with open(path, encoding="utf-8", newline="") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.rstrip("\r\n")
            m = INSERT_RE.match(line)
            if m:
                table = m.group(1)
                names = [c.strip().strip("`") for c in m.group(2).split(",")]
                if cols.setdefault(table, names) != names:
                    raise ValueError(f"baris {lineno}: daftar kolom {table} berubah")
                rows.setdefault(table, [])
                continue
            if table is None:
                continue
            if not line.startswith("\t("):
                raise ValueError(f"baris {lineno}: diharapkan tuple data, dapat {line[:80]!r}")
            vals, rest = parse_tuple(line)
            if len(vals) != len(cols[table]):
                raise ValueError(f"baris {lineno}: {len(vals)} nilai, {table} punya {len(cols[table])} kolom")
            rows[table].append(vals)
            if rest == ";":
                table = None
            elif rest != ",":
                raise ValueError(f"baris {lineno}: akhir tuple tidak dikenal {rest!r}")
    if table is not None:
        raise ValueError(f"dump berakhir di tengah INSERT tabel {table}")
    out = {}
    for t, r in rows.items():
        df = pd.DataFrame(r, columns=cols[t])
        for c in DATE_COLS.get(t, []):
            df[c] = pd.to_datetime(df[c], format="%Y-%m-%d", errors="raise")
        out[t] = df
    return out


def run(cfg: dict) -> dict:
    raw = Path(cfg["paths"]["raw_id_pharmacy"])
    dest = Path(cfg["paths"]["interim"]) / "id_pharmacy"
    dest.mkdir(parents=True, exist_ok=True)
    tables = parse_dump(raw / "sales.sql")
    summary = {}
    for name, df in tables.items():
        df.to_parquet(dest / f"{name}.parquet", index=False)
        summary[name] = {"rows": len(df), "columns": list(df.columns)}
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    print(json.dumps(run(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
