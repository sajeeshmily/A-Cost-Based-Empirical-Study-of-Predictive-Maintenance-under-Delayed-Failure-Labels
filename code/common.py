"""Shared components: detectors, calibrated learners, cost model."""
import numpy as np
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold


# ---------------------------------------------------------------- detectors
class ADWIN:
    """ADWIN (Bifet & Gavalda 2007), window capped at max_n, checked every `every` inserts.

    Uses the variance-based Hoeffding bound from the original paper.
    """

    def __init__(self, delta=0.002, max_n=3000, min_sub=50, every=25):
        self.delta, self.max_n, self.min_sub, self.every = delta, max_n, min_sub, every
        self.w = []
        self.k = 0

    def update(self, x):
        self.w.append(float(x))
        if len(self.w) > self.max_n:
            self.w.pop(0)
        self.k += 1
        if self.k % self.every:
            return False
        n = len(self.w)
        if n < 2 * self.min_sub:
            return False
        a = np.asarray(self.w)
        cs = np.cumsum(a)
        ks = np.arange(self.min_sub, n - self.min_sub + 1)
        n0, n1 = ks, n - ks
        m0 = cs[ks - 1] / n0
        m1 = (cs[-1] - cs[ks - 1]) / n1
        m = 1.0 / (1.0 / n0 + 1.0 / n1)
        dprime = self.delta / np.log(n)
        var = a.var()
        eps = np.sqrt(2.0 / m * var * np.log(2.0 / dprime)) + 2.0 / (3.0 * m) * np.log(2.0 / dprime)
        diff = np.abs(m0 - m1) - eps
        if diff.max() > 0:
            cut = ks[int(np.argmax(diff))]
            self.w = self.w[cut:]
            return True
        return False


def ks_alarm(ref, cur, alpha=0.01):
    """Per-feature two-sample KS with Bonferroni correction."""
    k = ref.shape[1]
    pmin = min(ks_2samp(ref[:, j], cur[:, j]).pvalue for j in range(k))
    return pmin < alpha / k


# ---------------------------------------------------------------- learners
class CalibratedModel:
    def __init__(self, base, iso):
        self.base, self.iso = base, iso

    def predict(self, X):
        return self.iso.predict(self.base.predict_proba(X)[:, 1])


def fit_model(X, y, seed, learner="rf", calib=None):
    """Fit a base learner and isotonic calibration on out-of-sample scores.

    Returns None when the data are too thin to fit (fewer than 5 failures).
    """
    y = np.asarray(y)
    if y.sum() < 5 or (len(y) - y.sum()) < 5:
        return None
    if learner == "rf":
        rf = RandomForestClassifier(n_estimators=60, min_samples_leaf=2, oob_score=True,
                                    n_jobs=1, random_state=seed)
        rf.fit(X, y)
        s = rf.oob_decision_function_[:, 1]
        ok = ~np.isnan(s)
        if calib is not None:  # calibrate on the recent window only (reservoir rows excluded)
            ok &= calib
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(s[ok], y[ok])
        return CalibratedModel(rf, iso)
    # histogram gradient boosting: 3-fold out-of-fold scores for calibration
    s = np.zeros(len(y))
    skf = StratifiedKFold(3, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=seed)
        m.fit(X[tr], y[tr])
        s[te] = m.predict_proba(X[te])[:, 1]
    full = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=seed).fit(X, y)
    ok = np.ones(len(y), bool) if calib is None else calib
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(s[ok], y[ok])
    return CalibratedModel(full, iso)


# ---------------------------------------------------------------- cost
def decision_cost(p, y, thr, r, n_updates, cu):
    """Total cost in units of one inspection (C_I = 1)."""
    alarm = p > thr
    fn = int(((~alarm) & (y == 1)).sum())
    tp = int((alarm & (y == 1)).sum())
    fp = int((alarm & (y == 0)).sum())
    tc = fn * r + (tp + fp) * 1.0 + n_updates * cu
    never = y.sum() * r
    return dict(TC=tc, NC=tc / never if never else np.nan, FN=fn, TP=tp, FP=fp, U=n_updates)
