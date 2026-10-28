"""AI4I 2020 controlled-drift study: prequential replay with delayed labels."""
import sys, os, time, json, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from common import ADWIN, ks_alarm, fit_model, decision_cost

DATA = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".csv") else None
WARM, N, BATCH, CHECK = 2000, 10000, 100, 200
WIN, VAL, HEXP, COOL = 1500, 500, 2000, 500
SENS = [3, 4, 5, 6]  # air, process, speed, torque columns (label-free detector inputs)
R_GRID, CU_GRID = [5, 10, 20, 50, 100], [0, 1, 10, 50]
MAGS = {"S0": [0], "S1": [0.1, 0.2, 0.3], "S2": [0.1, 0.2, 0.3], "S3": [0.5, 1.0, 1.5],
        "S4": [0.1, 0.25, 0.5], "S5": [1, 2, 3], "S6": [0]}


# ---------------------------------------------------------------- stream
def load(path):
    a = pd.read_csv(path)
    return dict(
        typ=a["Type"].values, air=a["Air temperature [K]"].values.astype(float),
        proc=a["Process temperature [K]"].values.astype(float),
        speed=a["Rotational speed [rpm]"].values.astype(float),
        torque=a["Torque [Nm]"].values.astype(float), wear=a["Tool wear [min]"].values.astype(float),
        y0=a["Machine failure"].values.astype(int))


def build_stream(raw, scen, mag, seed):
    rng = np.random.default_rng(seed)
    order = np.concatenate([np.arange(b * 100, b * 100 + 100) for b in rng.permutation(N // 100)])
    g = {k: v[order].copy() for k, v in raw.items()}
    t = np.arange(N)
    thr = pd.Series(g["typ"]).map({"L": 11000, "M": 12000, "H": 13000}).values
    y = g["y0"].copy()
    true_dT = g["proc"] - g["air"]

    def osf(m):
        return g["y0"] | (g["wear"] * g["torque"] > thr * (1 - m)).astype(int)

    def hdf(dd):
        return g["y0"] | ((true_dT < 8.6 + dd) & (g["speed"] < 1380)).astype(int)

    cps = []
    obs = {k: g[k].copy() for k in ["air", "proc", "speed", "torque"]}
    if scen == "S1":
        y = np.where(t >= 5000, osf(mag), y); cps = [5000]
    elif scen == "S2":
        pr = np.clip((t - 5000) / 2000, 0, 1)
        y = np.where(rng.random(N) < pr, osf(mag), y); cps = [5000, 7000]
    elif scen == "S3":
        on = ((t >= 4000) & (t < 6000)) | (t >= 8000)
        y = np.where(on, hdf(mag), y); cps = [4000, 6000, 8000]
    elif scen in ("S4", "S6"):
        sig = 0.25 if scen == "S6" else mag
        for k in obs:
            sd = raw[k].std()
            obs[k] = np.where(t >= 5000, obs[k] + rng.normal(0, sig * sd, N), obs[k])
        if scen == "S6":
            y = np.where(t >= 5000, osf(0.2), y)
        cps = [5000]
    elif scen == "S5":
        obs["proc"] = np.where(t >= 5000, obs["proc"] + mag, obs["proc"]); cps = [5000]
    onehot = np.c_[g["typ"] == "L", g["typ"] == "M", g["typ"] == "H"].astype(float)
    dT = obs["proc"] - obs["air"]
    power = obs["torque"] * obs["speed"] * 2 * np.pi / 60
    X = np.c_[onehot, obs["air"], obs["proc"], obs["speed"], obs["torque"], g["wear"], dT, power]
    return X, y.astype(int), cps


# ---------------------------------------------------------------- policies
class Policy:
    discrete = True  # each update is a retrain + redeploy (costs C_U)

    def __init__(self, name, sim):
        self.name, self.sim = name, sim
        self.nU, self.upd_t, self.alarms = 0.0, [], []
        self.last = -10**9

    def init(self):
        s = self.sim
        self.model = fit_model(s.X[:WARM], s.y[:WARM], s.seed, s.learner)

    def predict(self, Xb):
        return self.model.predict(Xb)

    def see(self, lo, hi):
        pass

    def release(self, lo, hi, t):
        pass

    def retrain(self, idx, t, calib=None):
        m = fit_model(self.sim.X[idx], self.sim.y[idx], self.sim.seed + int(self.nU) + 1, self.sim.learner, calib)
        if m is not None:
            self.model, self.nU, self.last = m, self.nU + 1, t
            self.upd_t.append(t)
            return True
        return False


class Static(Policy):
    pass


class ShiftAug(Policy):
    """Level-1 baseline: static model trained on warm-up plus noisy and offset copies."""

    def init(self):
        s = self.sim
        X, y = s.X[:WARM], s.y[:WARM]
        rng = np.random.default_rng(s.seed)
        Xn = X.copy()
        for j in SENS:
            Xn[:, j] += rng.normal(0, 0.25 * X[:, j].std(), len(X))
        Xo = X.copy(); Xo[:, 4] += 2.0
        for Z in (Xn, Xo):
            Z[:, 7] = Z[:, 4] - Z[:, 3]; Z[:, 8] = Z[:, 6] * Z[:, 5] * 2 * np.pi / 60
        self.model = fit_model(np.r_[X, Xn, Xo], np.r_[y, y, y], s.seed, s.learner)


class Periodic(Policy):
    def release(self, lo, hi, t):
        if t - max(self.last, WARM) >= 1000:
            self.retrain(np.arange(max(0, hi - 2000), hi), t)
            self.last = t


class SupDetect(Policy):
    def init(self):
        super().init(); self.ad = ADWIN()

    def release(self, lo, hi, t):
        p = self.sim.prob[self.name]
        fired = False
        for i in range(max(lo, WARM), hi):
            fired |= self.ad.update((p[i] - self.sim.y[i]) ** 2)
        if fired:
            self.alarms.append(t)
            if t - self.last >= COOL:
                self.retrain(np.arange(max(0, hi - WIN), hi), t)


class LabelFree(Policy):
    def init(self):
        super().init()
        self.ref = self.sim.X[WARM - 500:WARM][:, SENS]
        self.seen = WARM

    def see(self, lo, hi):
        self.seen = hi
        if (hi - WARM) % CHECK == 0 and hi - WARM >= 500:
            cur = self.sim.X[hi - 500:hi][:, SENS]
            if ks_alarm(self.ref, cur):
                self.alarms.append(hi)
                self.ref = cur
                if hi - self.last >= COOL:
                    self.pending = True
                    return
        self.pending = getattr(self, "pending", False)

    def release(self, lo, hi, t):
        if getattr(self, "pending", False):
            self.retrain(np.arange(max(0, hi - WIN), hi), t)
            self.pending = False


class ARFLite(Policy):
    """Adaptive-Random-Forest-style ensemble: each tree has its own ADWIN and is replaced on drift."""
    K = 10

    def _tree(self, idx, seed):
        rng = np.random.default_rng(seed)
        b = rng.choice(idx, len(idx), replace=True)
        if self.sim.y[b].sum() < 2:
            return None
        return DecisionTreeClassifier(min_samples_leaf=2, max_features="sqrt", random_state=seed).fit(self.sim.X[b], self.sim.y[b])

    def init(self):
        s = self.sim
        self.trees = [self._tree(np.arange(WARM), s.seed * 100 + k) for k in range(self.K)]
        self.ads = [ADWIN() for _ in range(self.K)]
        self.tp = np.full((self.K, N), np.nan)
        self.cnt = 0

    def predict(self, Xb):
        P = np.array([tr.predict_proba(Xb)[:, 1] if tr.classes_.size == 2 else np.zeros(len(Xb)) for tr in self.trees])
        lo = self._lo; self.tp[:, lo:lo + len(Xb)] = P
        return P.mean(0)

    def release(self, lo, hi, t):
        y = self.sim.y
        for k in range(self.K):
            fired = False
            for i in range(max(lo, WARM), hi):
                fired |= self.ads[k].update((self.tp[k, i] - y[i]) ** 2)
            if fired:
                self.cnt += 1
                nt = self._tree(np.arange(max(0, hi - 1000), hi), self.sim.seed * 1000 + self.cnt)
                if nt is not None:
                    self.trees[k] = nt; self.ads[k] = ADWIN()
                    self.nU += 1.0 / self.K; self.upd_t.append(t)


class OnlineLR(Policy):
    discrete = False

    def init(self):
        s = self.sim
        self.sc = StandardScaler().fit(s.X[:WARM])
        self.m = SGDClassifier(loss="log_loss", alpha=1e-4, random_state=s.seed)
        Z = self.sc.transform(s.X[:WARM])
        for _ in range(5):
            self.m.partial_fit(Z, s.y[:WARM], classes=[0, 1])

    def predict(self, Xb):
        return self.m.predict_proba(self.sc.transform(Xb))[:, 1]

    def release(self, lo, hi, t):
        lo = max(lo, WARM)
        if hi > lo:
            self.m.partial_fit(self.sc.transform(self.sim.X[lo:hi]), self.sim.y[lo:hi])


class Oracle(Policy):
    def release(self, lo, hi, t):
        for c in self.sim.cps:
            for k in (500, 1500):
                key = (c, k)
                if hi >= c + k and key not in getattr(self, "done", set()):
                    self.done = getattr(self, "done", set()) | {key}
                    self.retrain(np.arange(max(c, hi - 3000), hi), t)


class Adapt(Policy):
    """ADAPT-PM: two detectors, m-consecutive confirmation, cost gate, failure reservoir."""

    def __init__(self, name, sim, m=2, gate=True, reservoir=True, oracle=False):
        super().__init__(name, sim)
        self.m, self.gate, self.res, self.oracle = m, gate, reservoir, oracle
        self.gate_log = []

    def init(self):
        super().init()
        self.ad = ADWIN(); self.ad_fired = False
        self.ref = self.sim.X[WARM - 500:WARM][:, SENS]
        self.consec, self.pending, self.done = 0, False, set()

    def see(self, lo, hi):
        if self.oracle:
            return
        if (hi - WARM) % CHECK == 0 and hi - WARM >= 500:
            cur = self.sim.X[hi - 500:hi][:, SENS]
            ks = ks_alarm(self.ref, cur)
            alarm = ks or self.ad_fired
            self.ad_fired = False
            self.consec = self.consec + 1 if alarm else 0
            if self.consec >= self.m:
                self.alarms.append(hi)
                self.consec = 0
                self.ref = cur
                if hi - self.last >= COOL:
                    self.pending = True

    def release(self, lo, hi, t):
        s = self.sim
        if self.oracle:
            for c in s.cps:
                for k in (500, 1500):
                    if hi >= c + k and (c, k) not in self.done:
                        self.done.add((c, k)); self.pending = True
        else:
            p = s.prob[self.name]
            for i in range(max(lo, WARM), hi):
                self.ad_fired |= self.ad.update((p[i] - s.y[i]) ** 2)
        if self.pending:
            self.pending = False
            self.try_update(hi, t)

    def _res(self, start):
        if not self.res:
            return np.array([], int)
        f = np.where(self.sim.y[:start] == 1)[0]
        return f[-300:]

    def try_update(self, hi, t):
        s = self.sim
        w0 = max(0, hi - WIN)
        win = np.arange(w0, hi)
        if self.gate:
            V = np.arange(hi - VAL, hi)
            if s.y[V].sum() < 3:
                self.gate_log.append((t, "too_few_failures")); return
            rs = self._res(w0)
            tr = np.r_[rs, np.arange(w0, hi - VAL)]
            cm = np.r_[np.zeros(len(rs), bool), np.ones(hi - VAL - w0, bool)]
            cand = fit_model(s.X[tr], s.y[tr], s.seed + 7 + len(self.gate_log), s.learner, cm)
            if cand is None:
                self.gate_log.append((t, "no_candidate")); return
            thr = 1.0 / s.r

            def c(mdl):
                a = mdl.predict(s.X[V]) > thr
                return ((~a) & (s.y[V] == 1)).sum() * s.r + a.sum()

            saving = (c(self.model) - c(cand)) / VAL * HEXP
            if saving <= s.cu:
                self.gate_log.append((t, "rejected")); return
            self.gate_log.append((t, "accepted"))
        rs = self._res(w0)
        self.retrain(np.r_[rs, win], t, np.r_[np.zeros(len(rs), bool), np.ones(len(win), bool)])


# ---------------------------------------------------------------- simulation
class Sim:
    def __init__(self, X, y, cps, d, seed, learner="rf", r=20, cu=1):
        self.X, self.y, self.cps, self.d, self.seed, self.learner, self.r, self.cu = X, y, cps, d, seed, learner, r, cu
        self.prob = {}

    def run(self, policies):
        for p in policies:
            p.init(); self.prob[p.name] = np.full(N, np.nan)
        rel = WARM
        for s in range(WARM, N, BATCH):
            e = min(s + BATCH, N)
            for p in policies:
                if isinstance(p, ARFLite):
                    p._lo = s
                self.prob[p.name][s:e] = p.predict(self.X[s:e])
                p.see(s, e)
            nh = max(WARM, e - self.d)
            if nh > rel:
                for p in policies:
                    p.release(rel, nh, e)
                rel = nh


def make_policies(sim, full=True):
    P = [Static("P1_static_cost", sim)]
    if full:
        P += [ShiftAug("P0b_shift_aug", sim), Periodic("P2_periodic", sim), SupDetect("P3_sup_detect", sim),
              LabelFree("P4_labelfree", sim)]
        if sim.learner == "rf":
            P += [ARFLite("P5a_arf_lite", sim), OnlineLR("P5b_online_lr", sim)]
    P += [Adapt("P6_adapt_pm", sim)]
    if full:
        P += [Adapt("A1_no_gate", sim, gate=False), Adapt("A2_m1", sim, m=1),
              Adapt("A3_no_reservoir", sim, reservoir=False), Adapt("A4_oracle_detect", sim, oracle=True)]
        if sim.cps:
            P += [Oracle("P7_oracle", sim)]
    return P


GATED = {"P6_adapt_pm", "A2_m1", "A3_no_reservoir", "A4_oracle_detect"}


def evaluate(sim, policies, meta):
    rows = []
    y = sim.y[WARM:]
    for p in policies:
        pr = sim.prob[p.name][WARM:]
        base = dict(meta, policy=p.name, prauc=average_precision_score(y, pr), n_upd=p.nU,
                    n_alarm=len(p.alarms), gate=json.dumps([g[1] for g in getattr(p, "gate_log", [])]))
        # detection delay: first update at or after the first change point
        if sim.cps:
            c0 = sim.cps[0]
            after = [u for u in p.upd_t if u >= c0]
            base["det_delay"] = (after[0] - c0) if after else np.nan
            base["false_upd"] = sum(u < c0 for u in p.upd_t)
        else:
            base["det_delay"], base["false_upd"] = np.nan, len(p.upd_t)
        nu = p.nU if p.discrete else 0
        rs = [sim.r] if p.name in GATED else R_GRID
        cus = [sim.cu] if p.name in GATED else CU_GRID
        for r in rs:
            for cu in cus:
                thr = 1.0 / r
                rows.append(dict(base, r=r, cu=cu, thr="1/r", **decision_cost(pr, y, thr, r, nu, cu)))
                if p.name == "P1_static_cost":
                    rows.append(dict(base, policy="P0_static_0.5", r=r, cu=cu, thr="0.5", **decision_cost(pr, y, 0.5, r, 0, cu)))
    return rows


def job(args):
    scen, mag, d, seed, learner, r, cu, full = args
    raw = load(DATA_PATH)
    X, y, cps = build_stream(raw, scen, mag, seed)
    sim = Sim(X, y, cps, d, seed, learner, r, cu)
    pols = make_policies(sim, full)
    t0 = time.time()
    sim.run(pols)
    meta = dict(scen=scen, mag=mag, d=d, seed=seed, learner=learner, gate_r=r, gate_cu=cu, full=full)
    rows = evaluate(sim, pols, meta)
    for rw in rows:
        rw["secs"] = time.time() - t0
    return rows


DATA_PATH = None

if __name__ == "__main__":
    import argparse
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("out"); ap.add_argument("--plan", default="main")
    ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    DATA_PATH = a.data
    jobs = []
    if a.plan == "main":
        for scen, mags in MAGS.items():
            for mag in mags:
                mid = mag == mags[len(mags) // 2]
                for d in ([0, 50, 250, 1000] if mid else [50]):
                    for sd in range(a.seeds):
                        jobs.append((scen, mag, d, sd, "rf", 20, 1, True))
    elif a.plan == "s6rest":
        for d in [250, 1000]:
            for sd in range(a.seeds):
                jobs.append(("S6", 0, d, sd, "rf", 20, 1, True))
    elif a.plan == "gate":  # ADAPT-PM under other cost settings (middle magnitude, d = 50)
        for scen, mags in MAGS.items():
            mag = mags[len(mags) // 2]
            for r, cu in [(5, 1), (10, 1), (50, 1), (100, 1), (20, 0), (20, 10), (20, 50)]:
                for sd in range(a.seeds):
                    jobs.append((scen, mag, 50, sd, "rf", r, cu, False))
    elif a.plan == "hgb":  # second learner, middle magnitude, d = 50
        for scen, mags in MAGS.items():
            mag = mags[len(mags) // 2]
            for sd in range(a.seeds):
                jobs.append((scen, mag, 50, sd, "hgb", 20, 1, True))
    if a.limit:
        jobs = jobs[:a.limit]

    def init(p):
        global DATA_PATH
        DATA_PATH = p

    t0 = time.time(); out = []
    with Pool(a.procs, initializer=init, initargs=(a.data,)) as pool:
        for i, rows in enumerate(pool.imap_unordered(job, jobs)):
            out += rows
            if (i + 1) % 20 == 0 or i + 1 == len(jobs):
                pd.DataFrame(out).to_pickle(a.out)
                print(f"{i+1}/{len(jobs)} jobs, {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(out).to_pickle(a.out)
