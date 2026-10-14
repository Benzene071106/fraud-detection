# Compare model variants x decision rules on VALIDATION (selection), then report TEST for the chosen ones.
# Reads models/sweep/{aux,<variant>_P}.npz written by src.sweep_nohdr. CPU only, ~seconds per variant.
# Usage: python -m src.eval_sweep [--test]
import itertools
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "sweep"
CLASSES = ["Benign", "BruteForce", "DoS", "DDoS", "Bot", "Web", "Infiltration"]
K = len(CLASSES)
BF, DOS, DDOS, BOT, WEB, INF = 1, 2, 3, 4, 5, 6
FPR_TARGET, REVIEW_BUDGET, BASE_RATE = 0.002, 0.01, 0.01
A = np.load(OUT / "aux.npz", allow_pickle=True)
if_thr = np.quantile(A["if_va"][A["y_va"] == 0], 1 - REVIEW_BUDGET)


def macro_f1_ex_inf(y, pred):
    cm = np.bincount(y * K + pred, minlength=K * K).reshape(K, K)
    cm[INF, :] = 0
    tp = np.diag(cm).astype(float)
    p = np.divide(tp, cm.sum(0), out=np.zeros(K), where=cm.sum(0) > 0)
    r = np.divide(tp, cm.sum(1), out=np.zeros(K), where=cm.sum(1) > 0)
    f = np.divide(2 * p * r, p + r, out=np.zeros(K), where=(p + r) > 0)
    return f[[i for i in range(K) if i != INF]].mean()


def decide(P, thr, rule, mode):
    """mode 'perclass' = Variant B; 'flood' = DoS+DDoS probabilities pooled before thresholding."""
    Q = P.copy()
    if mode == "flood":
        pool = P[:, DOS] + P[:, DDOS]
        top = np.where(P[:, DOS] >= P[:, DDOS], DOS, DDOS)
        Q[:, DOS] = np.where(top == DOS, pool, 0)
        Q[:, DDOS] = np.where(top == DDOS, pool, 0)
    masked = np.where(Q >= thr, Q, -1.0)
    masked[:, 0] = -0.5
    cls = masked.argmax(1)
    cls = np.where(rule & (cls == 0), BF, cls)
    alert = (cls != 0) & (cls != INF)
    return alert, np.where(alert, cls, 0), cls


GRID = {"BruteForce": [0.5, 0.9, 0.95], "DoS": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95], "DDoS": [0.6, 0.7, 0.8, 0.9, 0.95],
        "Bot": [0.7, 0.9, 0.95], "Web": [0.8, 0.9, 0.95, 0.98]}


FLOOD_GRID = {**GRID, "DoS": [0.9, 0.95, 0.97, 0.98, 0.99, 0.995], "DDoS": [0.9, 0.95, 0.97, 0.98, 0.99, 0.995]}


def tune(P, y, rule, mode):
    best = None
    grid = FLOOD_GRID if mode == "flood" else GRID
    for combo in itertools.product(*grid.values()):
        thr = np.array([0.5, *combo, 0.9])
        alert, acls, _ = decide(P, thr, rule, mode)
        fpr = alert[y == 0].mean()
        if fpr > FPR_TARGET:
            continue
        f1 = macro_f1_ex_inf(y, acls)
        if best is None or (f1, -fpr) > (best[0], -best[1]):
            best = (f1, fpr, thr)
    return best


def tune_inf(P, y, ifs, thr, rule, mode):
    """Infiltration REVIEW threshold: most Infiltration caught with REVIEW FPR <= 1.5% (validation)."""
    best = None
    for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
        th = thr.copy(); th[INF] = t
        alert, _, cls = decide(P, th, rule, mode)
        review = ~alert & ((ifs >= if_thr) | (cls == INF))
        rf, inf = review[y == 0].mean(), (alert | review)[y == INF].mean()
        print(f"      inf_thr={t}: Infiltration A+R {inf:.2%}  REVIEW FPR {rf:.2%}")
        if rf <= 0.015 and (best is None or inf > best[1]):
            best = (t, inf, rf)
    return best


def report(P, y, ifs, rule, thr, mode):
    alert, acls, cls = decide(P, thr, rule, mode)
    review = ~alert & ((ifs >= if_thr) | (cls == INF))
    fpr, rec = alert[y == 0].mean(), alert[y != 0].mean()
    row = {"alert_fpr%": 100 * fpr, "review_fpr%": 100 * review[y == 0].mean(),
           "prec@1%": rec * BASE_RATE / (rec * BASE_RATE + fpr * (1 - BASE_RATE)),
           "detect_A+R%": 100 * (alert | review)[y != 0].mean(), "macroF1_exInf": macro_f1_ex_inf(y, acls)}
    for k, c in enumerate(CLASSES[1:], 1):
        row[c + "%"] = 100 * (alert | review)[y == k].mean()
    return row


rows, chosen = [], {}
for pf in sorted(OUT.glob("*_P.npz")):
    name = pf.name[:-6]
    P = np.load(pf)
    for mode in ["perclass", "flood"]:
        if len(sys.argv) > 1 and sys.argv[1] not in ("--test",) and name not in sys.argv[1].split(","):
            continue
        print(f"{name}/{mode}")
        b = tune(P["P_va"], A["y_va"], A["rule_va"], mode)
        if b is None:
            print(f"{name}/{mode}: no setting meets FPR target"); continue
        thr = b[2]
        ti = tune_inf(P["P_va"], A["y_va"], A["if_va"], thr, A["rule_va"], mode)
        if ti: thr[INF] = ti[0]
        r = report(P["P_va"], A["y_va"], A["if_va"], A["rule_va"], thr, mode)
        rows.append({"variant": name, "mode": mode, **r, "thr": np.round(thr, 2).tolist()})
        chosen[(name, mode)] = thr

res = pd.DataFrame(rows).sort_values("macroF1_exInf", ascending=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print("\n=== VALIDATION (selection) ===")
print(res.drop(columns="thr").round(4).to_string(index=False))
print(res[["variant", "mode", "thr"]].to_string(index=False))

if "--test" in sys.argv:
    print("\n=== TEST (read once, for the report) ===")
    trows = []
    for (name, mode), thr in chosen.items():
        P = np.load(OUT / f"{name}_P.npz")
        trows.append({"variant": name, "mode": mode,
                      **report(P["P_te"], A["y_te"], A["if_te"], A["rule_te"], thr, mode)})
    print(pd.DataFrame(trows).round(4).to_string(index=False))
