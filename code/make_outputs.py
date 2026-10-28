"""Publication figures (PNG 300 dpi + SVG) and a tables workbook from the stored results."""
import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, O = sys.argv[1], sys.argv[2]
os.makedirs(f"{O}/figures", exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
BLUE, GREY, DARK = "#2f6fdb", "#9a9a94", "#555555"

A = pd.read_csv(f"{R}/A_nc_by_scenario_d50.csv")
B = pd.read_csv(f"{R}/B_nc_by_delay.csv")
MID = {"S0": 0, "S1": 0.2, "S2": 0.2, "S3": 1.0, "S4": 0.25, "S5": 2, "S6": 0}
NAMES = {"S1": "S1 abrupt concept", "S2": "S2 gradual concept", "S3": "S3 recurring concept", "S6": "S6 concept + noise",
         "S4": "S4 sensor noise", "S5": "S5 sensor offset", "S0": "S0 stationary"}
ORDER = ["S1", "S2", "S3", "S6", "S4", "S5", "S0"]


def save(fig, name):
    fig.savefig(f"{O}/figures/{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{O}/figures/{name}.svg", bbox_inches="tight")
    plt.close(fig)


# Figure 2: cost by scenario
fig, ax = plt.subplots(figsize=(7, 4))
for i, sc in enumerate(ORDER):
    g = A[A.scen == sc].set_index("policy")
    y = len(ORDER) - 1 - i
    ax.plot(g.loc["P1_static_cost", "NC"], y, "o", mfc="none", mec=GREY, ms=7, label="Static, cost-optimal (P1)" if i == 0 else None)
    ax.plot(g.loc["P2_periodic", "NC"], y + 0.2, "s", color=GREY, ms=6, label="Periodic retraining (P2)" if i == 0 else None)
    if "P7_oracle" in g.index:
        ax.plot(g.loc["P7_oracle", "NC"], y, "|", color=DARK, ms=12, mew=2, label="Oracle (P7)" if i == 0 else None)
    ax.errorbar(g.loc["P6_adapt_pm", "NC"], y - 0.2, xerr=[[g.loc["P6_adapt_pm", "NC"] - g.loc["P6_adapt_pm", "lo"]], [g.loc["P6_adapt_pm", "hi"] - g.loc["P6_adapt_pm", "NC"]]],
                fmt="o", color=BLUE, ms=6, capsize=0, label="ADAPT-PM (P6), 95% CI" if i == 0 else None)
ax.set_yticks(range(len(ORDER)), [NAMES[s] for s in ORDER[::-1]])
ax.axhline(2.5, color="#dddddd", lw=0.8); ax.axhline(0.5, color="#dddddd", lw=0.8)
ax.set_xlim(0.1, 0.5); ax.set_xlabel("Normalised cost (lower is better)")
ax.set_title("Adaptation halves cost under real drift; ADAPT-PM ties periodic retraining", loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=4)
save(fig, "fig2_cost_by_scenario")

# Figure 3: cost against delay
real = B[B.scen.isin(["S1", "S2", "S3", "S6"])].groupby("d").mean(numeric_only=True)
fig, ax = plt.subplots(figsize=(6, 3.8))
lab = {"P1_static_cost": "Static, cost-optimal (P1)", "P3_sup_detect": "Supervised detector (P3)", "P2_periodic": "Periodic (P2)",
       "P6_adapt_pm": "ADAPT-PM (P6)", "P4_labelfree": "Label-free detector (P4)", "P7_oracle": "Oracle (P7)"}
xs = range(len(real.index))
for p, l in lab.items():
    st = dict(color=BLUE, lw=2.2, marker="o") if p == "P6_adapt_pm" else dict(color=DARK, ls="--", lw=1) if p == "P1_static_cost" else dict(color=GREY, lw=1.2, marker=".", ls=":" if p == "P7_oracle" else "-")
    ax.plot(xs, real[p].values, label=l, **st)
ax.set_xticks(list(xs), [f"{int(d):,}" for d in real.index]); ax.set_xlabel("Label delay (records)"); ax.set_ylabel("Normalised cost")
ax.set_title("Adaptation keeps most of its gain at a 1,000-record delay", loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=7.5, ncol=2)
save(fig, "fig3_cost_by_delay")

# Figure 4: ablations
grp = {"S1": "Real drift", "S2": "Real drift", "S3": "Real drift", "S6": "Real drift", "S4": "Input drift", "S5": "Input drift", "S0": "No drift"}
A["g"] = A.scen.map(grp)
abl = ["P6_adapt_pm", "A1_no_gate", "A2_m1", "A3_no_reservoir", "A4_oracle_detect", "P7_oracle"]
al = {"P6_adapt_pm": "ADAPT-PM", "A1_no_gate": "No cost gate", "A2_m1": "m = 1", "A3_no_reservoir": "No reservoir", "A4_oracle_detect": "Oracle timing", "P7_oracle": "Oracle"}
T = A[A.policy.isin(abl)].groupby(["policy", "g"]).NC.mean().unstack()[["Real drift", "Input drift", "No drift"]].reindex(abl)
fig, ax = plt.subplots(figsize=(6.5, 3.4))
w = 0.13
for i, p in enumerate(abl):
    hatch = {"A1_no_gate": "//", "A2_m1": "..", "A4_oracle_detect": "xx", "P7_oracle": "--"}.get(p, "")
    b = ax.bar(np.arange(3) + (i - 2.5) * w, T.loc[p].values, w, color=BLUE if p == "A3_no_reservoir" else "#d6d6d1" if p != "P6_adapt_pm" else DARK, edgecolor="#666666", lw=0.5, hatch=hatch, label=al[p])
    ax.bar_label(b, fmt="%.2f", fontsize=5.5, padding=1)
ax.set_xticks(range(3), T.columns); ax.set_ylabel("Normalised cost"); ax.set_ylim(0, 0.4)
ax.set_title("Removing the failure reservoir lowers cost in every group", loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=7.5, ncol=3)
save(fig, "fig4_ablations")

# Figure 5: update-cost sweep (r = 20, d = 50)
G = pd.read_csv(f"{R}/G_nc_by_cost_posthoc.csv")
F = pd.read_csv(f"{R}/F_payoff_adapt_vs_static.csv")
fig, axes = plt.subplots(1, 4, figsize=(9, 2.8), sharey=True)
for ax, sc in zip(axes, ["S1", "S3", "S4", "S0"]):
    g = G[(G.scen == sc) & (G.r == 20)].set_index("cu")
    f = F[(F.scen == sc) & (F.r == 20)].set_index("cu").sort_index()
    cus = [0, 1, 10, 50]
    ax.plot(range(4), [g.loc[c, "P1_static_cost"] for c in cus], "--", color=DARK, lw=1, label="Static (P1)")
    ax.plot(range(4), [g.loc[c, "P2_periodic"] for c in cus], "-", color=GREY, marker=".", label="Periodic (P2)")
    ax.plot(range(4), [g.loc[c, "P4_labelfree"] for c in cus], ":", color=GREY, marker=".", label="Label-free (P4)")
    ax.plot(range(4), [f.loc[c, "P6"] for c in cus], "-", color=BLUE, marker="o", lw=2, label="ADAPT-PM (P6)")
    ax.set_xticks(range(4), cus); ax.set_title(NAMES[sc], fontsize=9); ax.set_xlabel("Update cost C_U")
axes[0].set_ylabel("Normalised cost"); axes[0].legend(frameon=False, fontsize=7)
fig.suptitle("The cost gate pays when updates are expensive (r = 20, d = 50)", x=0.05, y=1.06, ha="left", fontsize=10)
save(fig, "fig5_update_cost")

# Figure 6: MetroPT-3
M = pd.read_pickle(f"{R}/metro_eval.pkl")
m = M[(M.task == "detect") & (M.H == "2h") & (M.d == "24h") & (M.cu == 10)]
pols = ["static", "periodic", "labelfree", "adapt", "rf_static", "rf_periodic", "rf_adapt"]
pn = {"static": "IF static", "periodic": "IF weekly refit", "labelfree": "IF KS refit", "adapt": "IF ADAPT-PM",
      "rf_static": "RF after 1st leak", "rf_periodic": "RF weekly", "rf_adapt": "RF ADAPT-PM"}
fig, ax = plt.subplots(figsize=(6.5, 3.2))
for j, r in enumerate([20, 100, 500]):
    v = m[m.r == r].groupby("policy").NC.mean().reindex(pols)
    ax.bar(np.arange(7) + (j - 1) * 0.27, v.values, 0.27, color=[GREY, "#7da2e8", BLUE][j], label=f"r = {r}")
ax.axhline(1, color=DARK, ls="--", lw=1); ax.text(6.6, 1.03, "never alarm", ha="right", fontsize=7.5, color=DARK)
ax.set_xticks(range(7), [pn[p] for p in pols], rotation=25, ha="right"); ax.set_ylabel("Normalised cost"); ax.set_ylim(0, 4.2)
ax.set_title("MetroPT-3: only models that learn released failures beat doing nothing", loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=8)
save(fig, "fig6_metropt3")

# Tables workbook
with pd.ExcelWriter(f"{O}/Tables_ADAPT-PM.xlsx") as xw:
    A.drop(columns="g").round(4).to_excel(xw, sheet_name="T5_by_scenario_d50", index=False)
    B.round(4).to_excel(xw, sheet_name="Delay_by_scenario", index=False)
    pd.read_csv(f"{R}/C_oracle_decomposition.csv").round(4).to_excel(xw, sheet_name="Oracle_decomposition", index=False)
    T.round(4).reset_index().to_excel(xw, sheet_name="T6_ablations", index=False)
    pd.read_csv(f"{R}/D_nc_by_magnitude.csv").round(4).to_excel(xw, sheet_name="Magnitude_d50", index=False)
    pd.read_csv(f"{R}/E_detectors.csv").round(2).to_excel(xw, sheet_name="Detector_behaviour", index=False)
    F.round(4).to_excel(xw, sheet_name="Cost_sweep_ADAPT_vs_static", index=False)
    G.round(4).to_excel(xw, sheet_name="T7_cost_posthoc_all", index=False)
    mm = M[M.cu == 10].groupby(["task", "H", "d", "r", "policy"]).agg(NC=("NC", "mean"), caught=("caught", "mean"), fa_per30=("fa_per30", "mean"), updates=("upd", "mean")).round(3).reset_index()
    mm.to_excel(xw, sheet_name="T8_MetroPT3", index=False)
    import json
    fr = json.load(open(f"{R}/stats.json"))["friedman"]
    rk = pd.Series(fr["mean_rank"], name="mean_rank").round(3).reset_index().rename(columns={"index": "policy"})
    rk["note"] = ""
    rk.loc[0, "note"] = f"Friedman chi2 = {fr['chi2']:.1f}, p = {fr['p']:.2e}, k = {fr['k']}, N = {fr['N']}, Nemenyi CD = {fr['CD']:.2f}"
    rk.to_excel(xw, sheet_name="Friedman_ranks", index=False)
print(os.listdir(f"{O}/figures"))
