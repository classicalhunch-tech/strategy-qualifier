"""A compact Deflated Sharpe Ratio estimate for per-trade returns."""
from __future__ import annotations

import math
import numpy as np


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_ppf(p: float) -> float:
    # Acklam inverse-normal approximation
    if not 0 < p < 1:
        raise ValueError("p must be between 0 and 1")
    a = [-39.6968302866538, 220.946098424521, -275.928510446969,
         138.357751867269, -30.6647980661472, 2.50662827745924]
    b = [-54.4760987982241, 161.585836858041, -155.698979859887,
         66.8013118877197, -13.2806815528857]
    c = [-0.00778489400243029, -0.322396458041136, -2.40075827716184,
         -2.54973253934373, 4.37466414146497, 2.93816398269878]
    d = [0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742]
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > 1-plow:
        q = math.sqrt(-2 * math.log(1-p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def deflated_sharpe(returns, n_trials: int = 1) -> dict:
    """Estimate DSR using the null Sharpe sampling variance 1/(T-1).

    Returns DSR in [0,1] when estimable. This is a screening statistic, not a
    pass/fail oracle; dependence and trial correlation require extra care.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    result = {"n": n, "n_trials": n_trials, "sharpe": float("nan"), "dsr": float("nan")}
    if n < 3:
        return result
    sd = float(x.std(ddof=1))
    if not math.isfinite(sd) or sd <= 0:
        return result
    sr = float(x.mean() / sd)
    z = (x - x.mean()) / x.std(ddof=0)
    skew = float(np.mean(z**3))
    excess_kurt = float(np.mean(z**4) - 3.0)
    variance = (1 - skew*sr + ((excess_kurt + 2.0)/4.0)*sr*sr) / (n-1)
    se = math.sqrt(variance) if variance > 0 else float("nan")
    expected_max_z = 0.0 if n_trials == 1 else (
        (1-0.5772156649015329) * _normal_ppf(1 - 1/n_trials)
        + 0.5772156649015329 * _normal_ppf(1 - 1/(n_trials*math.e))
    )
    sr0 = math.sqrt(1/(n-1)) * expected_max_z
    dsr = _normal_cdf((sr-sr0)/se) if math.isfinite(se) and se > 0 else float("nan")
    result.update(sharpe=sr, se=se, sr0=sr0, dsr=dsr)
    return result
