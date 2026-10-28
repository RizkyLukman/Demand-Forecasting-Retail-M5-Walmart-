"""Eksperimen te_item_mean jendela terbaru (ditunda dari Fase 6 §8.6), 2 toko × 4 fold CV.

Latar: `te_item_mean` = rata-rata penjualan item sepanjang riwayat train (bisa ±5 tahun),
fitur terpenting model (SHAP 42%). Akibatnya model lambat menyesuaikan diri: item yang tumbuh
diramal terlalu rendah, item yang turun terlalu tinggi (Fase 6 §8.5).

Rancangan (ditetapkan sebelum melihat hasil):

| Varian | Isi |
|---|---|
| base | konfigurasi final saat ini, dihitung ulang (harus sama dengan 0,69025 di tuning) |
| add_365 | + te_item_mean_365 (rata-rata 365 hari terakhir sampai cutoff) |
| add_182 | + te_item_mean_182 |
| replace_365 | te_item_mean diganti te_item_mean_365 |

- Toko CA_3 + WI_2, fold cv1-cv4, parameter `model.xgb_point` (sama dengan tuning).
- tau = noise seed dari tuning (`reports/data/xgb_tuning.json → noise.tau`, 0,00705).
- Aturan menerima: varian dengan WRMSSE rata-rata terendah diterima jika lebih baik dari
  base > tau DAN lebih baik di >= 3 dari 4 fold. Lalu dikonfirmasi di 10 toko × 4 fold
  dengan aturan yang sama terhadap WRMSSE 10 toko saat ini (0,58194).
- Bias per kelompok pertumbuhan dilaporkan sebagai pendukung, tidak dipakai untuk memutuskan.

Cara pakai:
    python -m retailfc.te_recent_experiment --config config.yaml
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from retailfc.io import load_config
from retailfc.xgb_final import model_params
from retailfc.xgb_tuning import FOLDS, Runner, accept

VARIANTS = {
    "add_365": {"te_recent_windows": (365,)},
    "add_182": {"te_recent_windows": (182,)},
    "replace_365": {"te_recent_windows": (365,), "te_drop_long": True},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rep = Path(cfg["paths"]["reports"]) / "data"
    tau = json.loads((rep / "xgb_tuning.json").read_text(encoding="utf-8"))["noise"]["tau"]
    run = Runner(cfg, rep / "te_recent_experiment.json")
    params, opt = model_params(cfg["model"]["xgb_point"])

    base = run.evaluate("base", params, opt)
    results = {name: run.evaluate(name, params, replace(opt, **v)) for name, v in VARIANTS.items()}
    best = min(results, key=lambda k: results[k]["wrmsse_mean"])
    ok = accept(results[best], base, tau)
    run.res["decision"] = {
        "tau": tau, "base": base["wrmsse_mean"],
        "candidates": {k: v["wrmsse_mean"] for k, v in results.items()},
        "best": best, "best_folds_better": sum(results[best]["per_fold"][f] < base["per_fold"][f] for f in FOLDS),
        "accepted": ok,
    }
    run.save()
    print(f"terbaik {best} ({results[best]['wrmsse_mean']:.5f}) vs base {base['wrmsse_mean']:.5f}: "
          f"{'DITERIMA' if ok else 'ditolak'}")
    for name, r in {"base": base, **results}.items():
        print(f"{name:12s} " + " ".join(f"{k}={v:+.3f}" for k, v in r["growth_bias"].items()))


if __name__ == "__main__":
    main()
