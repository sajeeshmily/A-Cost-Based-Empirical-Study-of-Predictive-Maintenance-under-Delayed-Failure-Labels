"""Event-level evaluation of MetroPT-3 runs, with day-block bootstrap intervals."""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import metro_sim as M

R = sys.argv[1]
runs = pickle.load(open(f"{R}/metro_runs.pkl", "rb"))
START = np.datetime64("2020-04-01")
rng = np.random.default_rng(0)
cache = {}


def data(task, H):
    if (task, H) not in cache:
        cache[(task, H)] = M.Data(f"{R}/metro_windows.pkl", H, task)
    return cache[(task, H)]


def assess(D, idx, task, H):
    al = np.zeros(len(D.t), bool); al[idx] = True
    al &= D.t >= START
    H = pd.Timedelta(H)
    ev = []
    for fs, fe in M.FAIL:
        a, b = (fs - H, fs) if task == "predict" else (fs, fs + pd.Timedelta("2h"))
        win = (D.t >= np.datetime64(a)) & (D.t < np.datetime64(b))
        infail = (D.t >= np.datetime64(fs)) & (D.t <= np.datetime64(fe))
        first = D.t[infail & al]
        ev.append(dict(day=np.datetime64(fs.normalize()), caught=bool(al[win].any()),
                       delay_min=float((first[0] - np.datetime64(fs)) / np.timedelta64(1, "m")) if len(first) else np.nan))
    fa = M.episodes(list(pd.DatetimeIndex(D.t[al & ~D.related])))
    fa_days = [np.datetime64(pd.Timestamp(e[0]).normalize()) for e in fa]
    return ev, fa_days


def nc(ev, fa_days, upd_days, r, cu):
    missed = sum(not e["caught"] for e in ev)
    caught = len(ev) - missed
    return (missed * r + caught + len(fa_days) + len(upd_days) * cu) / (len(ev) * r)


def boot(ev, fa_days, upd_days, r, cu, n=1000):
    days = pd.date_range("2020-04-01", "2020-08-31", freq="D").values.astype("datetime64[D]")
    nd = len(days)
    ev_by = {d: [] for d in days}; fa_by = {d: 0 for d in days}; up_by = {d: 0 for d in days}
    for e in ev:
        ev_by[e["day"].astype("datetime64[D]")].append(e)
    for d in fa_days:
        fa_by[d.astype("datetime64[D]")] = fa_by.get(d.astype("datetime64[D]"), 0) + 1
    for d in upd_days:
        k = d.astype("datetime64[D]")
        if k in up_by:
            up_by[k] += 1
    out = []
    for _ in range(n):
        starts = rng.integers(0, nd - 7, nd // 7 + 1)
        pick = np.concatenate([np.arange(s, s + 7) for s in starts])[:nd]
        E = [e for i in pick for e in ev_by[days[i]]]
        if not E:
            continue
        missed = sum(not e["caught"] for e in E)
        tc = missed * r + (len(E) - missed) + sum(fa_by[days[i]] for i in pick) + cu * sum(up_by[days[i]] for i in pick)
        out.append(tc / (len(E) * r))
    return np.quantile(out, [0.025, 0.975])


rows = []
RIND = {"static", "periodic", "labelfree"}  # policies whose behaviour does not depend on r
for run in runs:
    D = data(run["task"], run["H"])
    for pol, idx in run["alarms"].items():
        ev, fa = assess(D, idx, run["task"], run["H"])
        upd = [np.datetime64(pd.Timestamp(u)) for u in run["upd"][pol]]
        rs = [20, 100, 500] if pol in RIND else [run["r"]]
        for r in rs:
            for cu in [0, 10]:
                lo, hi = boot(ev, fa, upd, r, cu, 400) if (run["seed"] == 0 and cu == 10) else (np.nan, np.nan)
                rows.append(dict(task=run["task"], H=run["H"], d=run["d"], seed=run["seed"], r=r, cu=cu, policy=pol,
                                 NC=nc(ev, fa, upd, r, cu), lo=lo, hi=hi, caught=sum(e["caught"] for e in ev),
                                 detected_any=sum(not np.isnan(e["delay_min"]) for e in ev),
                                 median_delay_min=np.nanmedian([e["delay_min"] for e in ev]) if any(not np.isnan(e["delay_min"]) for e in ev) else np.nan,
                                 fa=len(fa), fa_per30=len(fa) / 153 * 30, upd=len(upd),
                                 events=",".join("1" if e["caught"] else "0" for e in ev),
                                 gate=" ".join(run["gate"].get(pol, []))))
df = pd.DataFrame(rows)
df.to_pickle(f"{R}/metro_eval.pkl")
pd.set_option("display.width", 250)
for task in ["detect", "predict"]:
    g = df[(df.task == task) & (df.H == "2h") & (df.cu == 10)]
    t = g.groupby(["d", "r", "policy"]).agg(NC=("NC", "mean"), caught=("caught", "mean"), fa=("fa", "mean"),
                                           upd=("upd", "mean"), lo=("lo", "mean"), hi=("hi", "mean")).round(3)
    print(task); print(t.to_string())
