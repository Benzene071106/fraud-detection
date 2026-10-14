# Experiments (one-off analyses)

These scripts produced results that are cited in the report (most outputs are in `reports/`). They are not part of
the running system. Run from the project root, e.g. `python -m src.experiments.cross_day`.

| Script | What it answered |
|---|---|
| `explore.py` | First look at the CSE-CIC-IDS2018 days, label counts, constant columns |
| `label_conflicts.py` | Identical feature vectors with different labels (Infiltration ↔ Benign twins) |
| `cross_day.py` | Train on some days, test on others: does the model generalise across days? |
| `loao_gpu.py` | Leave-one-attack-out: can XGBoost flag an attack family it never saw? |
| `hdr_ablation.py` | Header-length ablation: DoS 99.9 % → 81 % without the OS-fingerprint feature |
| `explain_v1.py`, `explain_xgb.py` | SHAP feature importance of the early / main XGBoost models |
| `fp_patterns.py` | What the remaining benign false alarms look like |
| `train_iforest.py` | Isolation Forest v1 (superseded by `src/iforest_tuning.py` → v2) |
| `hybrid_v1.py` | First hybrid (XGBoost + Rule R1 v1 + IF v1), superseded by `src/hybrid_v2.py` |
| `hybrid_loao.py`, `hybrid_final.py` | Hybrid detector under leave-one-attack-out and final hybrid evaluation |
| `iforest_operating_points.py` | Isolation Forest review budget vs detection trade-off |
| `prep_2017_web.py` | CIC-IDS2017 web-attack day as extra training data (w2017 model; did not help) |
| `nf_*.py` | NF-UQ-NIDS (NetFlow) experiments: per-site Isolation Forest vs transferred supervised model |

Models used only by these experiments were moved to `archive/cleanup_2026-10-14/models/` (not in git). Retrain or
move them back into `models/` before re-running the corresponding script.
