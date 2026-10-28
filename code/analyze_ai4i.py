"""Statistics for the AI4I study."""
import sys, json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare

R = sys.argv[1]  # results dir
main = pd.read_pickle(f"{R}/ai4i_main.pkl")
MID = {"S0": 0, "S1": 0.2, "S2": 0.2, "S3": 1.0, "S4": 0.25, "S5": 2, "S6": 0}
SCEN = ["S0", "S1", "S2", "S3", "S4", "S5", "S6"]
MAINP = ["P0_static_0.5", "P0b_shift_aug", "P1_static_cost", "P2_periodic", "P3_sup_detect", "P4_labelfree",
         "P5a_arf_lite", "P5b_online_lr", "P6_adapt_pm", "P7_oracle"]
rng = np.random.default_rng(0)


def boot_ci(x, n=2000):
    x = np.asarray(x)
    m = rng.choice(x, (n, len(x))).mean(1)
    return np.quantile(m, [0.025, 0.975])


def cliffs(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return ((a[:, None] < b[None, :]).sum() - (a[:, None] > b[None, :]).sum()) / (len(a) * len(b))


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0
    for i, k in enumerate(o):
        run = max(run, (m - i) * p[k]); adj[k] = min(1, run)
    return adj


def sel(df, **kw):
    for k, v in kw.items():
        df = df[df[k] == v]
    return df


base = main[(main.r == 20) & (main.cu == 1)]
mid = base[base.apply(lambda r: r.mag == MID[r.scen], axis=1)]
out = {}

# ---- Table A: NC by scenario and policy (mid magnitude, d = 50)
rows = []
for sc in SCEN:
    d50 = sel(mid, scen=sc, d=50)
    p1 = sel(d50, policy="P1_static_cost").set_index("seed")["NC"]
    for pol in MAINP + ["A1_no_gate", "A2_m1", "A3_no_reservoir", "A4_oracle_detect"]:
        x = sel(d50, policy=pol).set_index("seed")["NC"]
        if x.empty:
            continue
        lo, hi = boot_ci(x)
        diff = (x - p1.loc[x.index])
        try:
            pv = wilcoxon(x.values, p1.loc[x.index].values).pvalue if (diff != 0).any() else 1.0
        except ValueError:
            pv = 1.0
        rows.append(dict(scen=sc, policy=pol, NC=x.mean(), lo=lo, hi=hi, dNC=diff.mean(), p=pv,
                         cliff=cliffs(x.values, p1.loc[x.index].values),
                         prauc=sel(d50, policy=pol)["prauc"].mean(), upd=sel(d50, policy=pol)["n_upd"].mean(),
                         FN=sel(d50, policy=pol)["FN"].mean(), FP=sel(d50, policy=pol)["FP"].mean()))
A = pd.DataFrame(rows)
A["p_holm"] = np.nan
for sc in SCEN:
    m = (A.scen == sc) & (A.policy != "P1_static_cost")
    A.loc[m, "p_holm"] = holm(A.loc[m, "p"].values)
A.to_csv(f"{R}/A_nc_by_scenario_d50.csv", index=False)

# ---- Table B: label delay (mid magnitude), mean NC across seeds; and ΔNC vs P1
B = mid.groupby(["scen", "d", "policy"])["NC"].mean().unstack("policy")
B.to_csv(f"{R}/B_nc_by_delay.csv")

# ---- Table C: oracle decomposition at each delay (drift scenarios only)
crow = []
for (sc, d), g in mid[mid.scen != "S0"].groupby(["scen", "d"]):
    m = g.groupby("policy")["NC"].mean()
    crow.append(dict(scen=sc, d=d, P6=m.get("P6_adapt_pm"), A4=m.get("A4_oracle_detect"), P7=m.get("P7_oracle"),
                     gap_total=m.get("P6_adapt_pm") - m.get("P7_oracle"),
                     detection_part=m.get("P6_adapt_pm") - m.get("A4_oracle_detect"),
                     update_rule_part=m.get("A4_oracle_detect") - m.get("P7_oracle")))
pd.DataFrame(crow).to_csv(f"{R}/C_oracle_decomposition.csv", index=False)

# ---- Table D: magnitude sweep at d = 50
D = base[base.d == 50].groupby(["scen", "mag", "policy"])["NC"].mean().unstack("policy").reindex(columns=MAINP)
D.to_csv(f"{R}/D_nc_by_magnitude.csv")

# ---- Table E: detector behaviour (mid, d = 50)
E = sel(mid, d=50).groupby(["scen", "policy"])[["det_delay", "false_upd", "n_upd", "n_alarm"]].mean()
E.to_csv(f"{R}/E_detectors.csv")

# ---- Friedman + Nemenyi over all cells (scenario x magnitude x delay), main policies (no oracle)
FP = [p for p in MAINP if p != "P7_oracle"]
cell = base.groupby(["scen", "mag", "d", "policy"])["NC"].mean().unstack("policy")[FP].dropna()
ranks = cell.rank(axis=1)
stat, pf = friedmanchisquare(*[cell[c] for c in FP])
q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268}
k, N = len(FP), len(cell)
CD = q05[k] * np.sqrt(k * (k + 1) / (6 * N))
out["friedman"] = dict(chi2=float(stat), p=float(pf), k=k, N=N, CD=float(CD), mean_rank=ranks.mean().sort_values().to_dict())

# ---- Pay-off over cost settings: r sweep (cu = 1) and cu sweep (r = 20) at d = 50, mid magnitude
gate = pd.read_pickle(f"{R}/ai4i_gate.pkl") if __import__("os").path.exists(f"{R}/ai4i_gate.pkl") else None
prow = []
if gate is not None:
    for sc in SCEN:
        for (r, cu), g in gate[(gate.scen == sc)].groupby(["gate_r", "gate_cu"]):
            g6 = g[(g.policy == "P6_adapt_pm") & (g.r == r) & (g.cu == cu)].set_index("seed")["NC"]
            g1 = g[(g.policy == "P1_static_cost") & (g.r == r) & (g.cu == cu)].set_index("seed")["NC"]
            m2 = sel(mid, scen=sc, d=50)
            best_other = {}
            for pol in ["P2_periodic", "P3_sup_detect", "P4_labelfree"]:
                best_other[pol] = sel(m2, policy=pol).pipe(lambda x: x)  # placeholder
            prow.append(dict(scen=sc, r=r, cu=cu, P6=g6.mean(), P1=g1.mean(), dNC=(g6 - g1).mean(),
                             share_seeds_better=float(((g6 - g1) < 0).mean())))
    # add r = 20, cu = 1 from main
    for sc in SCEN:
        m2 = sel(mid, scen=sc, d=50)
        g6 = sel(m2, policy="P6_adapt_pm").set_index("seed")["NC"]; g1 = sel(m2, policy="P1_static_cost").set_index("seed")["NC"]
        prow.append(dict(scen=sc, r=20, cu=1, P6=g6.mean(), P1=g1.mean(), dNC=(g6 - g1).mean(), share_seeds_better=float(((g6 - g1) < 0).mean())))
    # r-independent policies across cost settings, from main (they were re-costed post hoc)
    allr = main[main.apply(lambda r: r.mag == MID[r.scen], axis=1) & (main.d == 50)]
    Q = allr.groupby(["scen", "r", "cu", "policy"])["NC"].mean().unstack("policy")
    Q.to_csv(f"{R}/G_nc_by_cost_posthoc.csv")
pd.DataFrame(prow).to_csv(f"{R}/F_payoff_adapt_vs_static.csv", index=False)

# ---- second learner
if __import__("os").path.exists(f"{R}/ai4i_hgb.pkl"):
    h = pd.read_pickle(f"{R}/ai4i_hgb.pkl")
    h = h[(h.r == 20) & (h.cu == 1)]
    H = h.groupby(["scen", "policy"])["NC"].mean().unstack("policy")
    H.to_csv(f"{R}/H_hgb.csv")

json.dump(out, open(f"{R}/stats.json", "w"), indent=1, default=float)
pd.set_option("display.width", 250)
print(A.pivot(index="policy", columns="scen", values="NC").round(3))
print(json.dumps(out, indent=1, default=float))
