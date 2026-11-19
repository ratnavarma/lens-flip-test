"""Thresholds and agreement metrics against Varma's labels. "?" labels are excluded before anything here.

Positive class = conveyed (label y). A rule predicts conveyed when cosine >= tau.

- choose_tau_kappa: the single tau that maximises Cohen's kappa on the labelled pairs (embedding mode).
- choose_band: tau_high and tau_low for the two-threshold rule (embedding+judge mode, and context in both):
  cosine >= tau_high is conveyed at >= 90% precision, cosine <= tau_low is not conveyed at >= 90% precision.
- judge_passes: the fixed pass rule for the local judge on the labelled borderline pairs.
"""

from __future__ import annotations

from typing import Any, Sequence

PRECISION_TARGET = 0.90
MIN_SUPPORT = 5                # labelled pairs a 90%-precision side needs before it counts
JUDGE_MIN_KAPPA = 0.6
JUDGE_MIN_ACCURACY = 0.80
JUDGE_MIN_LABELLED = 20        # fewer labelled borderline pairs than this cannot show a pass

Item = tuple[float, bool]      # (cosine, truth: conveyed)


def kappa(pred: Sequence[bool], truth: Sequence[bool]) -> float | None:
    """Cohen's kappa for two binary raters. None when there are no pairs. 0.0 when chance agreement is 1."""
    n = len(pred)
    if not n:
        return None
    po = sum(p == t for p, t in zip(pred, truth)) / n
    pp, pt = sum(pred) / n, sum(truth) / n
    pe = pp * pt + (1 - pp) * (1 - pt)
    return 0.0 if pe >= 1.0 else (po - pe) / (1 - pe)


def metrics(pred: Sequence[bool], truth: Sequence[bool]) -> dict[str, Any]:
    n = len(pred)
    tp = sum(p and t for p, t in zip(pred, truth))
    fp = sum(p and not t for p, t in zip(pred, truth))
    fn = sum(t and not p for p, t in zip(pred, truth))
    tn = n - tp - fp - fn

    def r(a: int, b: int) -> float | None:
        return a / b if b else None

    return {"n": n, "agreement": r(tp + tn, n), "precision": r(tp, tp + fp), "recall": r(tp, tp + fn),
            "specificity": r(tn, tn + fp), "kappa": kappa(pred, truth), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def rule_metrics(items: Sequence[Item], tau: float) -> dict[str, Any]:
    return metrics([c >= tau for c, _ in items], [t for _, t in items])


def choose_tau_kappa(items: Sequence[Item]) -> dict[str, Any]:
    """tau over the observed cosines that maximises kappa. Ties go to the lowest tau. In-sample."""
    cands = sorted({c for c, _ in items})
    best: tuple[float, float] | None = None
    for t in cands:
        k = kappa([c >= t for c, _ in items], [y for _, y in items])
        if k is not None and (best is None or k > best[0] + 1e-12):
            best = (k, t)
    if best is None:
        return {"tau": None, "reason": "no labelled pairs"}
    out = {"tau": best[1], "in_sample": True, "rule": "conveyed when cosine >= tau"}
    out.update(rule_metrics(items, best[1]))
    return out


def choose_band(items: Sequence[Item], target: float = PRECISION_TARGET, min_support: int = MIN_SUPPORT
                ) -> dict[str, Any]:
    """The two 90%-precision thresholds. A side with no qualifying threshold is None (nothing is decided there)."""
    cands = sorted({c for c, _ in items})
    high = None
    for t in reversed(cands):                      # from the top down: keep the lowest tau that still qualifies
        sel = [y for c, y in items if c >= t]
        if len(sel) >= min_support and sum(sel) / len(sel) >= target:
            high = t
    low = None
    for t in cands:                                # from the bottom up: keep the highest tau that still qualifies
        if high is not None and t >= high:
            break
        sel = [y for c, y in items if c <= t]
        if len(sel) >= min_support and (len(sel) - sum(sel)) / len(sel) >= target:
            low = t
    out: dict[str, Any] = {"target_precision": target, "min_support": min_support,
                           "tau_high": high, "tau_low": low}
    if high is not None:
        sel = [y for c, y in items if c >= high]
        out["high_support"] = {"n": len(sel), "precision_conveyed": sum(sel) / len(sel)}
    if low is not None:
        sel = [y for c, y in items if c <= low]
        out["low_support"] = {"n": len(sel), "precision_absent": (len(sel) - sum(sel)) / len(sel)}
    return out


def in_band(c: float, tau_low: float | None, tau_high: float | None) -> bool:
    """Borderline: not decided by the embedding alone. With no threshold on a side, that side decides nothing."""
    if tau_high is not None and c >= tau_high:
        return False
    if tau_low is not None and c <= tau_low:
        return False
    return True


def judge_passes(pred: Sequence[bool | None], truth: Sequence[bool]) -> dict[str, Any]:
    """The fixed rule: kappa >= 0.6 and accuracy >= 80% against the labels on the labelled borderline pairs.

    A judge answer that failed to parse (None) counts as wrong. At least JUDGE_MIN_LABELLED pairs are needed."""
    n = len(truth)
    fixed = [(not t) if p is None else p for p, t in zip(pred, truth)]
    m = metrics(fixed, list(truth))
    fails = sum(p is None for p in pred)
    reasons = []
    if n < JUDGE_MIN_LABELLED:
        reasons.append(f"only {n} labelled borderline pairs, at least {JUDGE_MIN_LABELLED} needed")
    if m["kappa"] is None or m["kappa"] < JUDGE_MIN_KAPPA:
        reasons.append(f"kappa {m['kappa'] if m['kappa'] is None else round(m['kappa'], 4)} below {JUDGE_MIN_KAPPA}")
    if m["agreement"] is None or m["agreement"] < JUDGE_MIN_ACCURACY:
        reasons.append(f"accuracy {m['agreement'] if m['agreement'] is None else round(m['agreement'], 4)} "
                       f"below {JUDGE_MIN_ACCURACY}")
    return {"passed": not reasons, "n": n, "parse_failures": fails, "accuracy": m["agreement"],
            "kappa": m["kappa"], "precision": m["precision"], "recall": m["recall"], "tp": m["tp"],
            "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
            "rule": {"kappa_min": JUDGE_MIN_KAPPA, "accuracy_min": JUDGE_MIN_ACCURACY,
                     "labelled_min": JUDGE_MIN_LABELLED, "fixed": "in advance, Varma 2026-10-21"},
            "reasons": reasons}
