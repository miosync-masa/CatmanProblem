from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "02_Data" / "Raw"
READY_DIR = ROOT / "02_Data" / "Analysis_Ready"
RESULTS_DIR = ROOT / "04_Results"
FIGURES_DIR = RESULTS_DIR / "Figures"

CONDITION_LABELS = {
    "C01": "Baseline",
    "C02": "Body",
    "C03": "Memory",
    "C04": "Function",
    "C05": "Self-ID uncertain",
    "C06": "Transition",
    "C07": "Body + transition",
    "C08": "Memory + transition",
    "A09": "Self-denial anchor",
    "A10": "Parallel anchor",
}

JUDGMENT_ITEMS = [
    "j_identity",
    "j_likeness",
    "j_relationship",
    "j_obligation",
    "j_treatment",
]

MANIPULATION_ITEMS = ["pc_sub", "pc_info", "pc_func", "pc_self", "pc_tran"]

ITEM_LABELS = {
    "j_identity": "Identity",
    "j_likeness": "Likeness",
    "j_relationship": "Relationship",
    "j_obligation": "Obligation",
    "j_treatment": "Treatment",
    "pc_sub": "Body continuity",
    "pc_info": "Memory continuity",
    "pc_func": "Functional continuity",
    "pc_self": "Self-continuity",
    "pc_tran": "Transition continuity",
    "k_copy": "Copy-likeness",
    "k_term": "Termination",
    "k_coex": "Coexistence",
    "j_ab": "Pre-X = surviving human",
    "j_ac": "Pre-X = generated cat",
    "j_bc": "Surviving human = generated cat",
    "j_excl": "Identity exclusivity",
    "v_epistemic": "Epistemic veto",
    "v_normative": "Normative veto",
}


def ensure_dirs() -> None:
    READY_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _betacf(a: float, b: float, x: float) -> float:
    max_iter, eps, fpmin = 200, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = fpmin if abs(d) < fpmin else d
        c = 1.0 + aa / c
        c = fpmin if abs(c) < fpmin else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = fpmin if abs(d) < fpmin else d
        c = 1.0 + aa / c
        c = fpmin if abs(c) < fpmin else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_cdf(t_value: float, df: int) -> float:
    x = df / (df + t_value * t_value)
    ib = _betai(df / 2.0, 0.5, x)
    return 1.0 - 0.5 * ib if t_value >= 0 else 0.5 * ib


def t_ppf(probability: float, df: int) -> float:
    lo, hi = -50.0, 50.0
    for _ in range(120):
        mid = (lo + hi) / 2.0
        if t_cdf(mid, df) < probability:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def paired_summary(differences: Iterable[float]) -> dict[str, float | int]:
    d = np.asarray(list(differences), dtype=float)
    d = d[np.isfinite(d)]
    n = int(d.size)
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else math.nan
    se = sd / math.sqrt(n) if n > 1 else math.nan
    tcrit = t_ppf(0.975, n - 1) if n > 1 else math.nan
    ci_low, ci_high = mean - tcrit * se, mean + tcrit * se
    if sd == 0:
        dz = math.copysign(math.inf, mean) if mean != 0 else 0.0
        t_value = dz
        p_value = 0.0 if mean != 0 else 1.0
    else:
        dz = mean / sd
        t_value = mean / se
        p_value = 2.0 * (1.0 - t_cdf(abs(t_value), n - 1))
    return {
        "n": n,
        "mean_difference": mean,
        "sd_difference": sd,
        "se_difference": se,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "dz": dz,
        "t": t_value,
        "df": n - 1,
        "p_ttest": max(0.0, min(1.0, p_value)),
    }


def paired_condition_effect(
    data: pd.DataFrame, condition: str, baseline: str, item: str
) -> dict[str, float | int | str]:
    wide = data[
        data["condition_id"].isin([baseline, condition]) & (data["item"] == item)
    ].pivot(index="participant_id", columns="condition_id", values="value")
    wide = wide.dropna(subset=[baseline, condition])
    out = paired_summary(wide[condition] - wide[baseline])
    out.update(
        {
            "condition_id": condition,
            "condition_label": CONDITION_LABELS[condition],
            "baseline_id": baseline,
            "item": item,
            "item_label": ITEM_LABELS[item],
            "baseline_mean": float(wide[baseline].mean()),
            "condition_mean": float(wide[condition].mean()),
        }
    )
    return out


def linear_contrast_effect(
    data: pd.DataFrame, coefficients: dict[str, float], item: str, label: str
) -> dict[str, float | int | str]:
    wide = data[
        data["condition_id"].isin(coefficients) & (data["item"] == item)
    ].pivot(index="participant_id", columns="condition_id", values="value")
    wide = wide.dropna(subset=list(coefficients))
    values = sum(wide[c] * weight for c, weight in coefficients.items())
    out = paired_summary(values)
    out.update({"contrast": label, "item": item, "item_label": ITEM_LABELS[item]})
    return out


def bh_fdr(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def profile_difference_matrix(
    data: pd.DataFrame,
    condition: str,
    baseline: str = "C01",
    items: list[str] | None = None,
) -> np.ndarray:
    items = items or JUDGMENT_ITEMS
    wide = data[
        data["condition_id"].isin([baseline, condition]) & data["item"].isin(items)
    ].pivot(index="participant_id", columns=["condition_id", "item"], values="value")
    required = [(baseline, item) for item in items] + [(condition, item) for item in items]
    wide = wide.dropna(subset=required)
    return np.column_stack(
        [
            (wide[(condition, item)] - wide[(baseline, item)]).to_numpy(dtype=float)
            for item in items
        ]
    )


def permutation_profile_test(
    differences: np.ndarray,
    permutations: int = 100_000,
    seed: int = 20260814,
    batch_size: int = 2_000,
) -> dict[str, float | int]:
    query_means = differences.mean(axis=0)
    observed = float(np.sum((query_means - query_means.mean()) ** 2))
    rng = np.random.default_rng(seed)
    exceedances = 0
    for start in range(0, permutations, batch_size):
        batch = min(batch_size, permutations - start)
        order = np.argsort(
            rng.random((batch, differences.shape[0], differences.shape[1])), axis=2
        )
        permuted = np.take_along_axis(
            np.broadcast_to(differences, (batch, *differences.shape)), order, axis=2
        )
        means = permuted.mean(axis=1)
        statistic = np.sum((means - means.mean(axis=1, keepdims=True)) ** 2, axis=1)
        exceedances += int(np.sum(statistic >= observed - 1e-15))
    p_value = (exceedances + 1) / (permutations + 1)
    return {
        "n": int(differences.shape[0]),
        "n_queries": int(differences.shape[1]),
        "statistic": observed,
        "permutations": permutations,
        "exceedances": exceedances,
        "p_permutation": p_value,
    }


def wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return centre - half, centre + half


def pearson_summary(x: Iterable[float], y: Iterable[float]) -> dict[str, float | int]:
    pair = pd.DataFrame({"x": list(x), "y": list(y)}).dropna()
    n = len(pair)
    r = float(pair["x"].corr(pair["y"]))
    if n <= 3 or not np.isfinite(r):
        return {"n": n, "r": r, "ci95_low": math.nan, "ci95_high": math.nan, "p": math.nan}
    clipped = min(0.999999999, max(-0.999999999, r))
    fisher = math.atanh(clipped)
    se = 1.0 / math.sqrt(n - 3)
    ci_low = math.tanh(fisher - 1.959963984540054 * se)
    ci_high = math.tanh(fisher + 1.959963984540054 * se)
    t_value = r * math.sqrt((n - 2) / max(1e-15, 1.0 - r * r))
    p_value = 2.0 * (1.0 - t_cdf(abs(t_value), n - 2))
    return {"n": n, "r": r, "ci95_low": ci_low, "ci95_high": ci_high, "p": p_value}


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.6f")

