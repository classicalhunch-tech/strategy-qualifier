"""Purged walk-forward (time-fold) analysis for dev-data exploration.

Reuses ols_cluster from tools/session_vs_outside.py. Its exact signature was not
visible when this was written, so the adapter `_fit` below assumes
    ols_cluster(y, X, clusters) -> (coef_array, se_array)
Adjust `_fit` in one place if yours differs.

Custom `fit` callables should accept (y, window, clusters=None). A callable that
only accepts (y, window) is still supported and is called without clusters.

    python tools/walk_forward.py --trades dev_trades.csv --returns-col net_r \
        --window-col in_window --n-folds 5 --label-horizon 5 --embargo-days 5
"""
from __future__ import annotations

import argparse
import inspect

import numpy as np
import pandas as pd


def purge_overlap_and_embargo(train_dates, test_dates, label_horizon: int = 5,
                              embargo_days: int = 5):
    """Return the train dates that survive purging and embargo.

    Drops train t where t + horizon >= test_start (label overlaps the test period,
    for t <= test_end), and train t in (test_end, test_end + embargo].
    """
    train = pd.DatetimeIndex(pd.to_datetime(list(train_dates)))
    test = pd.DatetimeIndex(pd.to_datetime(list(test_dates)))
    if len(test) == 0:
        return train
    t0, t1 = test.min(), test.max()
    h = pd.Timedelta(days=label_horizon)
    e = pd.Timedelta(days=embargo_days)
    overlap = (train + h >= t0) & (train <= t1)
    embargo = (train > t1) & (train <= t1 + e)
    return train[~(overlap | embargo)]


def make_folds(n: int, n_folds: int):
    """Contiguous index folds over time-sorted rows."""
    edges = np.linspace(0, n, n_folds + 1).astype(int)
    return [np.arange(edges[i], edges[i + 1]) for i in range(n_folds)]


def ols_cluster(y, X, clusters):
    """OLS with cluster-robust CR1 standard errors (numpy only)."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    try:
        inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return None
    beta = inv @ (X.T @ y)
    resid = y - X @ beta
    _, labels = np.unique(np.asarray(list(clusters), dtype=str), return_inverse=True)
    g = int(labels.max()) + 1
    if g < 2 or n <= k:
        return beta, np.full(k, float("nan"))
    meat = np.zeros((k, k))
    for cluster_id in range(g):
        mask = labels == cluster_id
        score = X[mask].T @ resid[mask]
        meat += np.outer(score, score)
    scale = (g / (g - 1)) * ((n - 1) / (n - k))
    variance = scale * (inv @ meat @ inv)
    return beta, np.sqrt(np.clip(np.diag(variance), 0.0, None))


def _fit(y, window, clusters=None):
    X = np.column_stack([np.ones(len(y)), window])
    if clusters is None:
        clusters = np.arange(len(y))  # each trade its own cluster
    res = ols_cluster(y, X, clusters)
    if res is None:  # singular design, e.g. every trade in the same group
        return float("nan"), float("nan")
    return res


def _accepts_clusters(fit) -> bool:
    try:
        params = inspect.signature(fit).parameters
    except (TypeError, ValueError):
        return True
    return "clusters" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def run(df: pd.DataFrame, returns_col="net_r", window_col="in_window",
        time_col="entry_time", n_folds=5, label_horizon=5, embargo_days=5,
        fit=None) -> pd.DataFrame:
    fit = fit or _fit
    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col])
    d = d.sort_values(time_col).reset_index(drop=True)
    rows = []
    for k, idx in enumerate(make_folds(len(d), n_folds), start=1):
        test = d.iloc[idx]
        if len(test) < 2:
            rows.append(dict(fold=k, n_train=0, n_test=len(test), coef=np.nan, se=np.nan,
                             t=np.nan, error="fewer than 2 trades in fold"))
            continue
        rest = d.drop(d.index[idx])
        keep = purge_overlap_and_embargo(rest[time_col], test[time_col],
                                         label_horizon, embargo_days)
        n_train = int(rest[time_col].isin(keep).sum())
        error = ""
        try:
            y = test[returns_col].to_numpy(float)
            w = test[window_col].to_numpy(float)
            if _accepts_clusters(fit):
                test_dates = test[time_col].dt.date.astype(str).to_numpy()
                coef, se = fit(y, w, clusters=test_dates)
            else:
                coef, se = fit(y, w)
            coef, se = float(np.atleast_1d(coef)[-1]), float(np.atleast_1d(se)[-1])
            t = coef / se if se and np.isfinite(se) else np.nan
        except Exception as exc:  # keep going, but say what went wrong
            coef = se = t = np.nan
            error = f"{type(exc).__name__}: {exc}"
        if not error and np.isnan(coef):
            error = "not estimable (window dummy has no variation in this fold)"
        rows.append(dict(fold=k, n_train=n_train, n_test=len(test), coef=coef, se=se, t=t,
                         error=error))
    return pd.DataFrame(rows)


def summarize(table: pd.DataFrame) -> dict:
    c = table["coef"].dropna()
    t = table["t"].dropna()
    return dict(mean_coef=float(c.mean()) if len(c) else np.nan,
                sd_coef=float(c.std(ddof=1)) if len(c) > 1 else np.nan,
                frac_positive=float((c > 0).mean()) if len(c) else np.nan,
                frac_abs_t_gt_2=float((t.abs() > 2).mean()) if len(t) else np.nan)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", required=True)
    ap.add_argument("--returns-col", default="net_r")
    ap.add_argument("--window-col", default="in_window")
    ap.add_argument("--time-col", default="entry_time")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--label-horizon", type=int, default=5)
    ap.add_argument("--embargo-days", type=int, default=5)
    a = ap.parse_args(argv)
    tbl = run(pd.read_csv(a.trades), a.returns_col, a.window_col, a.time_col,
              a.n_folds, a.label_horizon, a.embargo_days)
    print(tbl.to_string(index=False))
    s = summarize(tbl)
    print(f"\nmean coef {s['mean_coef']:.4f} | SD {s['sd_coef']:.4f} | "
          f"frac positive {s['frac_positive']:.2f} | frac |t|>2 {s['frac_abs_t_gt_2']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
