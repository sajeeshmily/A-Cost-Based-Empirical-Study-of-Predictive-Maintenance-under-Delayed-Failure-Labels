"""MetroPT-3: day-by-day replay with delayed labels; label-free and supervised policies."""
import sys, os, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from common import fit_model
from scipy.stats import ks_2samp

FAIL = [("2020-04-18 00:00", "2020-04-18 23:59"), ("2020-05-29 23:30", "2020-05-30 06:00"),
        ("2020-06-05 10:00", "2020-06-07 14:30"), ("2020-07-15 14:30", "2020-07-15 19:00")]
FAIL = [(pd.Timestamp(s), pd.Timestamp(e)) for s, e in FAIL]
START = pd.Timestamp("2020-04-01")
TRAIN_DAYS, VAL_DAYS, HEXP_DAYS, COOL_DAYS = 28, 7, 28, 7
MERGE = pd.Timedelta("1h")
def ks_d(a, b):
    return max(ks_2samp(a[:, j], b[:, j]).statistic for j in range(a.shape[1]))


def ks_threshold(ks, t, q=0.95):
    """Effect-size threshold: 95th percentile of KS distance between consecutive 3-day blocks in training data."""
    day = np.timedelta64(1, "D")
    t0 = t.min() + 6 * day
    vals = []
    while t0 <= t.max():
        a = ks[(t >= t0 - 6 * day) & (t < t0 - 3 * day)]; b = ks[(t >= t0 - 3 * day) & (t < t0)]
        if len(a) > 288 and len(b) > 288:
            vals.append(ks_d(a, b))
        t0 += day
    return np.quantile(vals, q)


KS_COLS = ["TP2_mean", "TP3_mean", "H1_mean", "DV_pressure_mean", "Reservoirs_mean", "Oil_temperature_mean", "Motor_current_mean"]


def episodes(times):
    """Group alarm times into episodes: a new episode starts after a 1 h silence."""
    eps = []
    for t in times:
        if eps and t - eps[-1][1] <= MERGE:
            eps[-1][1] = t
        else:
            eps.append([t, t])
    return eps


class Data:
    def __init__(self, path, H, task="predict"):
        F = pd.read_pickle(path)
        self.t = F.index.values.astype("datetime64[ns]")
        self.T = F.index
        self.cols = [c for c in F.columns if c not in ("n", "event", "in_fail")]
        self.X = F[self.cols].values.astype(float)
        self.ks = F[KS_COLS].values
        self.in_fail = F["in_fail"].values
        H = pd.Timedelta(H)
        self.H = H
        Data.H_ = H
        Data.task_ = task
        self.y = np.zeros(len(F), int)
        self.rel_base = self.T + H  # a window's label is known once its horizon has passed
        self.related = np.zeros(len(F), bool)  # pre-failure or in-failure windows
        for s, e in FAIL:
            pre = (self.T >= s - H) & (self.T < s) if task == "predict" else (self.T >= s) & (self.T <= e)
            self.y[pre] = 1
            rel = (self.T >= s - H) & (self.T <= e)
            self.related |= rel
            self.rel_base = self.rel_base.where(~rel, e)
        self.rel_base = self.rel_base.values.astype("datetime64[ns]")


def label_free_threshold(scores, times, per_days=7):
    """Lowest threshold whose alarm-episode rate on training data is at most one per `per_days` days."""
    span = (times.max() - times.min()) / np.timedelta64(1, "D")
    budget = max(1, int(span / per_days))
    for q in [0.99, 0.995, 0.998, 0.999, 0.9995, 0.9998, 0.9999, 1.0]:
        thr = np.quantile(scores, q)
        if len(episodes(list(pd.DatetimeIndex(times[scores > thr])))) <= budget:
            return thr
    return scores.max()


class IFModel:
    def __init__(self, X, times, seed):
        self.sc = StandardScaler().fit(X)
        self.f = IsolationForest(n_estimators=100, max_samples=512, random_state=seed).fit(self.sc.transform(X))
        s = self.score(X)
        self.thr = label_free_threshold(s, times)

    def score(self, X):
        return -self.f.score_samples(self.sc.transform(X))

    def alarm(self, X):
        return self.score(X) > self.thr


def ev_window(fs, fe):
    """Window in which an alarm counts as catching the event."""
    if Data.task_ == "predict":
        return fs - Data.H_, fs
    return fs, fs + pd.Timedelta("2h")


def event_cost(alarm, t, y_pre, r):
    """Event-level cost on a set of windows: r per missed failure, 1 per alarm episode outside failures."""
    cost = 0.0
    neg = alarm & (y_pre == 0)
    cost += len(episodes(list(pd.DatetimeIndex(t[neg]))))
    for fs, fe in FAIL:
        a, b = ev_window(fs, fe)
        pre = (t >= np.datetime64(a)) & (t < np.datetime64(b))
        if pre.any():
            cost += 1.0 if alarm[pre].any() else r
    return cost


class RFModel:
    def __init__(self, X, y, t, seed, r):
        from sklearn.ensemble import RandomForestClassifier
        self.rf = RandomForestClassifier(n_estimators=60, min_samples_leaf=2, oob_score=True, n_jobs=1,
                                         random_state=seed, class_weight="balanced_subsample").fit(X, y)
        oob = np.nan_to_num(self.rf.oob_decision_function_[:, 1])
        self.oob = oob
        # false-alarm budget on normal training windows (at most one alarm episode per week)
        self.thr = label_free_threshold(oob[y == 0], t[y == 0])

    def prob(self, X):
        return self.rf.predict_proba(X)[:, 1]

    def alarm(self, X):
        return self.prob(X) > self.thr


class Policy:
    def __init__(self, name, sim, kind):
        self.name, self.sim, self.kind = name, sim, kind
        self.nU, self.upd_t, self.alarms_t, self.gate_log = 0, [], [], []
        self.last = np.datetime64("1970-01-01")

    # normal training windows released by time `now`, within the last `days` days before `end`
    def normal_idx(self, now, days, end=None):
        D = self.sim.D
        end = now if end is None else end
        m = (D.rel_base <= now) & (~D.related) & (D.t < end) & (D.t >= end - np.timedelta64(days, "D"))
        return np.where(m)[0]

    def init(self):
        D = self.sim.D
        w = np.where(D.t < np.datetime64(START))[0]
        self.model = IFModel(D.X[w], D.t[w], self.sim.seed)
        self.ref = D.ks[w[-3 * 288:]]
        self.ks_thr = ks_threshold(D.ks[w], D.t[w])
        self.consec = 0
        self.sup = None

    def alarm(self, idx):
        D = self.sim.D
        if self.sup is not None:
            return self.sup.alarm(D.X[idx])
        return self.model.alarm(D.X[idx])

    def refit_if(self, now):
        idx = self.normal_idx(now, TRAIN_DAYS)
        if len(idx) < 288 * 7:
            return False
        self.model = IFModel(self.sim.D.X[idx], self.sim.D.t[idx], self.sim.seed + self.nU + 1)
        self.nU += 1; self.upd_t.append(now); self.last = now
        return True

    def sup_data(self, now):
        """Last 28 days of released normal windows plus a reservoir of every released failure-related window."""
        D = self.sim.D
        res = np.where((D.rel_base <= now) & D.related & ((~D.in_fail) | (Data.task_ == "detect")))[0]
        return np.union1d(self.normal_idx(now, TRAIN_DAYS), res)

    def fit_sup(self, now, force_model=None):
        D = self.sim.D
        idx = self.sup_data(now)
        if D.y[idx].sum() < 5:
            return False
        m = force_model or RFModel(D.X[idx], D.y[idx], D.t[idx], self.sim.seed + self.nU, self.sim.r)
        self.sup = m; self.nU += 1; self.upd_t.append(now); self.last = now
        self.n_events_seen = self.events_released(now)
        return True

    def events_released(self, now):
        D = self.sim.D
        return sum(int(((D.y == 1) & (D.rel_base <= now) & (D.t >= np.datetime64(fs - D.H)) & (D.t <= np.datetime64(fe))).any()) for fs, fe in FAIL)

    def rf_gate(self, now):
        """Gate: compare the deployed model (out-of-sample) with a candidate (out-of-bag) on windows released since the last update."""
        D, s = self.sim.D, self.sim
        if self.sup is None:
            return self.fit_sup(now)
        idx = self.sup_data(now)
        if D.y[idx].sum() < 5:
            return False
        cand = RFModel(D.X[idx], D.y[idx], D.t[idx], s.seed + 100 + len(self.gate_log), s.r)
        since = (D.rel_base[idx] > self.last) | (D.y[idx] == 1)
        if since.sum() < 288:
            self.gate_log.append("too_little_data"); return False
        S = idx[since]
        c_cur = event_cost(self.sup.alarm(D.X[S]), D.t[S], D.y[S], s.r)
        c_new = event_cost(cand.oob[since] > cand.thr, D.t[S], D.y[S], s.r)
        span = max(1.0, (D.t[S].max() - D.t[S].min()) / np.timedelta64(1, "D"))
        saving = (c_cur - c_new) * HEXP_DAYS / span
        if saving > s.cu:
            self.gate_log.append("accepted")
            return self.fit_sup(now, force_model=cand)
        self.gate_log.append("rejected"); self.last_gate = now
        return False

    def end_of_day(self, now, day_idx):
        k = self.kind
        D = self.sim.D
        if k == "periodic":
            if (now - self.last) >= np.timedelta64(7, "D"):
                self.refit_if(now) or setattr(self, "last", now)
        elif k in ("labelfree", "adapt", "adapt_m1"):
            cur = D.ks[(D.t >= now - np.timedelta64(3, "D")) & (D.t < now)]
            if len(cur) < 288:
                return
            alarm = ks_d(self.ref, cur) > self.ks_thr
            m = 2 if k == "adapt" else 1
            self.consec = self.consec + 1 if alarm else 0
            if self.consec >= m:
                self.alarms_t.append(now); self.consec = 0; self.ref = cur
                if (now - self.last) >= np.timedelta64(COOL_DAYS, "D"):
                    if k == "labelfree":
                        self.refit_if(now)
                    else:
                        self.gate(now)
        elif k == "rf_static":
            if self.sup is None:
                self.fit_sup(now)
        elif k == "rf_periodic":
            if self.sup is None or (now - self.last) >= np.timedelta64(7, "D"):
                self.fit_sup(now)
        elif k == "rf_adapt":
            # triggers: a newly released failure, or label-free drift confirmed on 2 consecutive days
            new_event = self.events_released(now) > getattr(self, "n_events_seen", 0)
            cur = D.ks[(D.t >= now - np.timedelta64(3, "D")) & (D.t < now)]
            drift = False
            if len(cur) >= 288:
                self.consec = self.consec + 1 if ks_d(self.ref, cur) > self.ks_thr else 0
                if self.consec >= 2:
                    drift = True; self.consec = 0; self.ref = cur; self.alarms_t.append(now)
            if new_event or (drift and (now - self.last) >= np.timedelta64(COOL_DAYS, "D")):
                self.rf_gate(now)
                self.n_events_seen = self.events_released(now)

    def gate(self, now):
        """Cost gate on the last 7 days of released windows."""
        D, s = self.sim.D, self.sim
        rel_end = now - s.dly
        vmask = (D.t >= rel_end - np.timedelta64(VAL_DAYS, "D")) & (D.t < rel_end) & (D.rel_base <= now)
        V = np.where(vmask)[0]
        tr = self.normal_idx(now, TRAIN_DAYS, end=rel_end - np.timedelta64(VAL_DAYS, "D"))
        if len(V) < 288 or len(tr) < 288 * 7:
            self.gate_log.append("too_little_data"); return
        cand = IFModel(D.X[tr], D.t[tr], s.seed + 50 + len(self.gate_log))

        def cost(a):
            at = pd.DatetimeIndex(D.t[V][a])
            eps = episodes(list(at))
            cost = 0.0
            for e0, e1 in eps:
                cost += 1.0  # each alarm episode is one inspection
            for fs, fe in FAIL:  # failures whose scoring window lies in V
                a0, b0 = ev_window(fs, fe)
                pre = (D.t[V] >= np.datetime64(a0)) & (D.t[V] < np.datetime64(b0))
                if pre.any() and not a[pre].any():
                    cost += s.r
            return cost

        saving = (cost(self.model.alarm(D.X[V])) - cost(cand.alarm(D.X[V]))) * HEXP_DAYS / VAL_DAYS
        if saving > s.cu:
            self.gate_log.append("accepted")
            self.model = cand; self.nU += 1; self.upd_t.append(now); self.last = now
        else:
            self.gate_log.append("rejected")


class Sim:
    def __init__(self, D, d, seed, r, cu):
        self.D, self.dly, self.seed, self.r, self.cu = D, np.timedelta64(pd.Timedelta(d)), seed, r, cu

    def run(self, pols):
        D = self.D
        alarms = {p.name: np.zeros(len(D.t), bool) for p in pols}
        for p in pols:
            p.init()
        days = pd.date_range(START, D.T.max().normalize() + pd.Timedelta("1D"), freq="D")
        for d0, d1 in zip(days[:-1], days[1:]):
            idx = np.where((D.t >= np.datetime64(d0)) & (D.t < np.datetime64(d1)))[0]
            for p in pols:
                if len(idx):
                    alarms[p.name][idx] = p.alarm(idx)
                # labels released by d1 are those with rel_base + delay <= d1
                p.end_of_day(np.datetime64(d1), idx)
        return alarms


def released_now_wrapper():
    pass


KINDS = ["static", "periodic", "labelfree", "adapt", "rf_static", "rf_periodic", "rf_adapt"]


def run_one(path, H, d, seed, r, cu, kinds=KINDS, task="predict"):
    D = Data(path, H, task)
    sim = Sim(D, d, seed, r, cu)
    # labels are visible to a policy at time `now` when rel_base + delay <= now: shift rel_base by the delay
    D.rel_base = D.rel_base + np.timedelta64(pd.Timedelta(d))
    pols = [Policy(k, sim, k) for k in kinds]
    t0 = time.time()
    al = sim.run(pols)
    return dict(task=task, H=H, d=d, seed=seed, r=r, cu=cu, secs=time.time() - t0,
                alarms={k: np.where(v)[0] for k, v in al.items()},
                upd={p.name: [str(u) for u in p.upd_t] for p in pols},
                gate={p.name: p.gate_log for p in pols},
                ks_alarms={p.name: len(p.alarms_t) for p in pols})


if __name__ == "__main__":
    path, out = sys.argv[1], sys.argv[2]
    plan = []
    for task in ["detect", "predict"]:
        for d in ["0h", "1h", "24h", "7D"]:
            for sd in range(3):
                plan.append((task, "2h", d, sd))
    for H in ["30min", "6h"]:
        for sd in range(3):
            plan.append(("predict", H, "24h", sd))
    res = []
    t0 = time.time()
    for i, (task, H, d, sd) in enumerate(plan):
        for r in [20, 100, 500]:
            kinds = KINDS if r == 100 else ["adapt", "rf_static", "rf_periodic", "rf_adapt"]  # r-dependent policies
            res.append(run_one(path, H, d, sd, r, 10, kinds, task))
        print(i + 1, len(plan), f"{time.time()-t0:.0f}s", flush=True)
        pickle.dump(res, open(out, "wb"))
