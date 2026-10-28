"""MetroPT-3: 5-minute window features and labels."""
import sys
import numpy as np, pandas as pd

src, out = sys.argv[1], sys.argv[2]
ANA = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]
DIG = ["COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level", "Caudal_impulses"]
FAIL = [("2020-04-18 00:00", "2020-04-18 23:59"), ("2020-05-29 23:30", "2020-05-30 06:00"),
        ("2020-06-05 10:00", "2020-06-07 14:30"), ("2020-07-15 14:30", "2020-07-15 19:00")]

m = pd.read_csv(src, usecols=["timestamp"] + ANA + DIG, parse_dates=["timestamp"]).sort_values("timestamp")
m["w"] = m["timestamp"].dt.floor("5min")
m["t"] = (m["timestamp"] - m["w"]).dt.total_seconds()
# a window must not span a gap > 10 min; drop windows touched by such a gap
gap = m["timestamp"].diff().dt.total_seconds() > 600
bad = set(m.loc[gap, "w"]) | set(m.loc[gap.shift(-1, fill_value=False), "w"])
g = m.groupby("w")
feats = {}
n = g.size()
for c in ANA:
    x = m[c]
    feats[f"{c}_mean"] = g[c].mean(); feats[f"{c}_std"] = g[c].std().fillna(0)
    feats[f"{c}_min"] = g[c].min(); feats[f"{c}_max"] = g[c].max()
    # least-squares slope against seconds within the window
    tx = (m["t"] * x).groupby(m["w"]).sum(); st = g["t"].sum(); sx = g[c].sum(); stt = (m["t"] ** 2).groupby(m["w"]).sum()
    den = n * stt - st ** 2
    feats[f"{c}_slope"] = ((n * tx - st * sx) / den.replace(0, np.nan)).fillna(0)
same = m["w"].eq(m["w"].shift())
for c in DIG:
    feats[f"{c}_share"] = g[c].mean()
    feats[f"{c}_switch"] = (m[c].diff().ne(0) & same).groupby(m["w"]).sum()
F = pd.DataFrame(feats)
F["n"] = n
F = F[(F["n"] >= 10) & (~F.index.isin(bad))].copy()
F.index.name = "w"
# trailing context (causal): sustained load is invisible inside a single 5-minute window
CTX = ["DV_eletric_share", "COMP_share", "Motor_current_mean", "TP2_mean", "Oil_temperature_mean", "Reservoirs_mean"]
for c in CTX:
    for win in ["1h", "3h"]:
        F[f"{c}_r{win}"] = F[c].rolling(win).mean()
# event bookkeeping
F["event"] = -1
F["in_fail"] = False
for k, (s, e) in enumerate(FAIL):
    s, e = pd.Timestamp(s), pd.Timestamp(e)
    F.loc[(F.index >= s) & (F.index <= e), "in_fail"] = True
    F.loc[(F.index >= s - pd.Timedelta("6h")) & (F.index <= e), "event"] = k
F.to_pickle(out)
print(F.shape, F.index.min(), F.index.max(), F["in_fail"].sum())
