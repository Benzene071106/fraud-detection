"""Engineered ratio features, computed from existing flow features (no header lengths, no fingerprints).
Shared by training (src.sweep_nohdr) and the API detector so both compute them identically."""
import numpy as np
import pandas as pd

RATIO_FEATURES = ["r_bwd_fwd_pkts", "r_bwd_fwd_bytes", "r_dur_per_pkt", "r_data_pkt_frac",
                  "r_ctrl_flags_per_pkt", "r_idle_active"]
RATIO_INPUTS = ["Total Fwd Packets", "Total Backward Packets", "Fwd Packets Length Total",
                "Bwd Packets Length Total", "Flow Duration", "Fwd Act Data Packets", "SYN Flag Count",
                "RST Flag Count", "FIN Flag Count", "Idle Mean", "Active Mean"]


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    f = {c: df[c].astype("float64").values for c in RATIO_INPUTS}
    pk = f["Total Fwd Packets"] + f["Total Backward Packets"]
    out = {
        "r_bwd_fwd_pkts": f["Total Backward Packets"] / (f["Total Fwd Packets"] + 1),
        "r_bwd_fwd_bytes": f["Bwd Packets Length Total"] / (f["Fwd Packets Length Total"] + 1),
        "r_dur_per_pkt": f["Flow Duration"] / (pk + 1),
        "r_data_pkt_frac": f["Fwd Act Data Packets"] / (f["Total Fwd Packets"] + 1),
        "r_ctrl_flags_per_pkt": (f["SYN Flag Count"] + f["RST Flag Count"] + f["FIN Flag Count"]) / (pk + 1),
        "r_idle_active": f["Idle Mean"] / (f["Active Mean"] + 1),
    }
    return df.assign(**{k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
                        for k, v in out.items()})
