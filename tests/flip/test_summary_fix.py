"""Summary fixes after the v3 merge report: protocol labels, provenance, explicit denominators, the
recoverability breakdown of residue flips, and a field-by-field numeric check of old against new summaries.
No network: every runner is a fake. The stored-results check skips when results/ is absent (it is git-ignored).

Run the stored-results check by hand:
    uv run python tests/flip/test_summary_fix.py results/headline-300-v3/summary_v3.pre-fix.json \
        results/headline-300-v3/summary_v3.json
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest

from lens.flip.headline import engine as E
from lens.flip.headline import merge as MG
from lens.flip.headline import metrics as HM
from lens.flip.headline import mock as M

ROOT = Path(__file__).resolve().parents[2]

# Paths (as dotted keys, "*" for any key) whose value is corrected on purpose: the protocol label.
CORRECTED = (("residue", "*", "power_rule", "protocol"), ("residue", "by_system", "*", "power_rule", "protocol"),
             ("config", "version"))


def _corrected(path: tuple[str, ...]) -> bool:
    for pat in CORRECTED:
        if len(pat) == len(path) and all(a in ("*", b) for a, b in zip(pat, path)):
            return True
    return False


def compare_numeric(old: Any, new: Any, path: tuple[str, ...] = ()) -> tuple[list[str], list[tuple[str, Any, Any]]]:
    """Walk `old` and compare every leaf with the leaf at the same path in `new`.

    Keys that exist only in `new` are ignored (new fields). A key of `old` missing from `new` is a difference.
    Numbers must be equal exactly (bools and ints included, NaN equal to NaN). The corrected protocol labels are
    returned as the second value instead of counted as differences.
    Returns (differences, corrections).
    """
    diffs: list[str] = []
    fixes: list[tuple[str, Any, Any]] = []
    here = ".".join(path)
    if isinstance(old, dict):
        if not isinstance(new, dict):
            return [f"{here}: dict became {type(new).__name__}"], fixes
        for k, v in old.items():
            if k not in new:
                diffs.append(f"{here}.{k}: missing in new")
                continue
            d, f = compare_numeric(v, new[k], (*path, k))
            diffs += d
            fixes += f
        return diffs, fixes
    if isinstance(old, list):
        if not isinstance(new, list) or len(old) != len(new):
            return [f"{here}: list differs in length or type"], fixes
        for i, (a, b) in enumerate(zip(old, new)):
            d, f = compare_numeric(a, b, (*path, str(i)))
            diffs += d
            fixes += f
        return diffs, fixes
    same = old == new or (isinstance(old, float) and isinstance(new, float) and math.isnan(old) and math.isnan(new))
    if not same:
        if _corrected(path):
            fixes.append((here, old, new))
        else:
            diffs.append(f"{here}: {old!r} -> {new!r}")
    return diffs, fixes


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

def _finish(d: Path, seed: int = 1) -> None:
    cfg = json.loads((d / "config.json").read_text())
    cfg["embedder"] = {"kind": "hash", "model": "hash-bow-256"}
    cfg["bootstrap"]["reps"] = 100
    cfg["h2b_draws"] = 50
    (d / "config.json").write_text(json.dumps(cfg))
    eng = E.Engine(d, check_protocol=False)
    model = M.MockModel(eng.cases, M.v4_calibration(), seed=seed)
    eng.runner = M.FakeBatchAPI(model.answer, M.engine_meta_fn(d))
    for _ in range(600):
        if eng.step(1e9) == "complete":
            break
    else:
        raise AssertionError("did not finish")
    eng.close()
    HM.write_report(d, 2)


@pytest.fixture(scope="module")
def merged(tmp_path_factory):
    root = tmp_path_factory.mktemp("fix")
    base, u2 = root / "base", root / "u2"
    E.init_run(base, 24, 5)
    E.init_run(u2, 4, 6, ("U2",), "u3")
    _finish(base)
    _finish(u2, seed=2)
    out = root / "merged"
    s = MG.merge_report(base, u2, "U2", out, 3)
    return {"root": root, "base": base, "u2": u2, "out": out, "summary": s,
            "disk": json.loads((out / "summary_v3.json").read_text())}


# ---------------------------------------------------------------------------
# 1. provenance
# ---------------------------------------------------------------------------

SIZES = {"H2a_n": 30, "H2b_n": 5, "H2c_pos": 25, "H2c_neg": 40}
THR = HM.power_thresholds({})


def test_power_report_carries_the_protocol_it_is_given():
    assert HM.power_report(SIZES, THR, 3)["protocol"] == 3
    assert HM.power_report(SIZES, THR, 2)["protocol"] == 2
    with pytest.raises(TypeError):
        HM.power_report(SIZES, THR)  # no default: a caller must say which protocol


def test_merged_summary_says_protocol_three_everywhere(merged):
    s = merged["disk"]
    assert s["protocol"] == 3 and s["config"]["version"] == "headline-v3"
    r = s["residue"]
    assert r["pooled"]["power_rule"]["protocol"] == 3
    assert {b["power_rule"]["protocol"] for b in r["by_system"].values()} == {3}
    assert s["provenance"]["protocol"] == 3
    # the run's own config.json is not rewritten
    assert json.loads((merged["base"] / "config.json").read_text())["version"] == "headline-v2"
    assert s["schema"] == HM.SUMMARY_SCHEMA == "lens.flip.summary/1"


def test_provenance_block_of_a_merge(merged):
    pv = merged["disk"]["provenance"]
    assert pv["protocol_file"] == "docs/flip-protocol-v3.md"
    assert pv["protocol_file_sha256"] == _sha(ROOT / "docs" / "flip-protocol-v3.md")
    assert pv["protocol_frozen_commit"] is None or len(pv["protocol_frozen_commit"]) == 40
    assert pv["merge"]["replaced_mechanism"] == "U2"
    assert pv["merge"]["base_dir"].endswith("base") and pv["merge"]["replacement_dir"].endswith("u2")
    assert [r["role"] for r in pv["runs"]] == ["base", "replacement"]
    for role, d in (("base", merged["base"]), ("replacement", merged["u2"])):
        run = next(r for r in pv["runs"] if r["role"] == role)
        files = next(v for k, v in pv["inputs_sha256"].items() if Path(k).name == d.name)
        assert files["files"]["config.json"] == _sha(d / "config.json")
        assert files["files"]["cases.jsonl"] == _sha(d / "cases.jsonl")
        assert files["state_cases"]["n"] == len(list((d / "state" / "cases").glob("*.json")))
        assert run["config_version"] == "headline-v2"
    # the same bytes on a second run
    again = merged["root"] / "again"
    MG.merge_report(merged["base"], merged["u2"], "U2", again, 3)
    assert (again / "summary_v3.json").read_bytes() == (merged["out"] / "summary_v3.json").read_bytes()


def test_single_run_report_has_provenance(merged):
    s = HM.write_report(merged["u2"], 3)
    pv = s["provenance"]
    assert pv["protocol"] == 3 and [r["role"] for r in pv["runs"]] == ["run"]
    assert s["config"]["version"] == "headline-v3"
    assert s["residue"]["pooled"]["power_rule"]["protocol"] == 3
    assert json.loads((merged["u2"] / "summary_v3.json").read_text())["provenance"] == json.loads(json.dumps(pv))


# ---------------------------------------------------------------------------
# 2. denominators
# ---------------------------------------------------------------------------

def test_denominator_fields_are_present_and_consistent(merged):
    s = merged["disk"]
    assert "rate_all" in s["field_notes"] and "alias" in s["field_notes"]["rate"]
    for name, c in s["cells"].items():
        hf = c["harmful_flip"]
        n_all, n_gated = hf["n_all"], hf["n_gated"]
        assert n_all == c["n"] and 0 < n_gated <= n_all
        for blk in (hf, hf["wrong_commit"], hf["abstention"], hf["parse_error"]):
            assert blk["n_all"] == n_all and blk["n_gated"] == n_gated
            assert blk["rate"] == blk["rate_all"]  # the alias
            assert blk["rate_all"]["value"] == pytest.approx(blk["count"] / n_all)
            assert blk["rate_gated"]["value"] == pytest.approx(blk["count"] / n_gated)
            assert blk["rate_all"]["sums"]["one"] == n_all and blk["rate_gated"]["sums"]["g"] == n_gated
        assert hf["count"] == sum(hf[k]["count"] for k in ("wrong_commit", "abstention", "parse_error"))


def test_tables_name_the_denominator(merged):
    text = (merged["out"] / "tables_v3.txt").read_text()
    head = next(ln for ln in text.splitlines() if ln.startswith("cell "))
    assert "wrong commit % (of gated n)" in head and "abstention % (of gated n)" in head
    assert "harmful % (of gated n)" in head
    cells = merged["disk"]["cells"]
    hf = next(iter(cells.values()))["harmful_flip"]
    line = next(ln for ln in text.splitlines() if ln.startswith("denominators per cell"))
    assert f"n_all {hf['n_all']}" in line and f"n_gated {hf['n_gated']}" in line
    assert text.splitlines().index(line) < text.splitlines().index(head)


# ---------------------------------------------------------------------------
# 3. recoverability
# ---------------------------------------------------------------------------

def test_recoverability_counts_sum_to_the_residue_flips(merged):
    r = merged["disk"]["residue"]
    for name, b in [("pooled", r["pooled"]), *r["by_system"].items()]:
        rc = b["recoverability"]
        assert rc["n_flips"] == b["n_residue_flips"], name
        assert sum(rc["counts"].values()) == b["n_residue_flips"], name
        assert rc["restored"]["count"] == rc["counts"]["restored_1_minimal"] + rc["counts"]["restored_cap_hit"]
        if rc["n_flips"]:
            assert sum(rc["shares"].values()) == pytest.approx(1.0)
        d = rc["not_recoverable_diagnostic"]
        assert d["descriptive_only"] is True and d["n"] == rc["counts"]["not_recoverable"]
        assert 0 <= d["b_equals_trunc"]["count"] <= d["n"] and 0 <= d["b_request_info"]["count"] <= d["n"]
        assert sum(d["b_actions"].values()) == d["n"]
    pooled = r["pooled"]["recoverability"]["counts"]
    for k in pooled:
        assert pooled[k] == sum(b["recoverability"]["counts"][k] for b in r["by_system"].values())
    # the old searches block agrees
    assert r["searches"]["n"] == r["pooled"]["n_residue_flips"]
    assert r["searches"]["status"].get("restored", 0) == pooled["restored_1_minimal"] + pooled["restored_cap_hit"]
    assert r["searches"]["cap_hits"] == pooled["restored_cap_hit"]


def test_recoverability_on_a_toy_table():
    def row(cid, system, flip, a_b, a_trunc, status):
        return {"case_id": cid, "system": system, "residue_flip": flip, "a_B": a_b, "a_trunc": a_trunc,
                "search_status": status}

    rows = [row("a", "s", True, "X", "X", "restored"), row("b", "s", True, "REQUEST_INFO", "X", "restored"),
            row("c", "s", True, "REQUEST_INFO", "REQUEST_INFO", "not_recoverable_by_insertion"),
            row("d", "s", True, "X", "Y", "not_recoverable_by_insertion"),
            row("e", "s", True, "Y", "Y", "not_recoverable_by_insertion"), row("f", "s", False, "Z", "Z", None)]
    recs = {("a", "s"): {"status": "restored", "hit_cap": False}, ("b", "s"): {"status": "restored", "hit_cap": True},
            ("c", "s"): {"status": "not_recoverable_by_insertion", "hit_cap": False},
            ("d", "s"): {"status": "not_recoverable_by_insertion", "hit_cap": False},
            ("e", "s"): {"status": "not_recoverable_by_insertion", "hit_cap": False}}
    rc = HM.recoverability(rows, recs)
    assert rc["counts"] == {"restored_1_minimal": 1, "restored_cap_hit": 1, "not_recoverable": 3, "other": 0}
    assert rc["shares"]["not_recoverable"] == pytest.approx(0.6)
    d = rc["not_recoverable_diagnostic"]
    assert d["n"] == 3 and d["b_equals_trunc"]["count"] == 2 and d["b_request_info"]["count"] == 1
    assert HM.recoverability([], {})["shares"]["restored_1_minimal"] is None


# ---------------------------------------------------------------------------
# 4. numeric equality of old and new
# ---------------------------------------------------------------------------

def test_compare_numeric_finds_changes_and_ignores_new_fields_and_corrected_labels():
    old = {"a": {"rate": {"value": 0.5, "ci95": [0.1, 0.9]}}, "config": {"version": "headline-v2", "n": 3},
           "residue": {"pooled": {"power_rule": {"protocol": 2, "H2a": {"n": 4}}}}}
    new = json.loads(json.dumps(old))
    new["a"]["rate_all"] = {"value": 0.5}  # a new field
    new["config"]["version"] = "headline-v3"
    new["residue"]["pooled"]["power_rule"]["protocol"] = 3
    diffs, fixes = compare_numeric(old, new)
    assert diffs == [] and len(fixes) == 2
    new["a"]["rate"]["ci95"][1] = 0.9000001
    new["residue"]["pooled"]["power_rule"]["H2a"]["n"] = 5
    del new["config"]["n"]
    diffs, _ = compare_numeric(old, new)
    assert len(diffs) == 3


def test_a_regenerated_merge_is_numerically_equal_to_its_earlier_output(merged, tmp_path):
    """Old summary (before the fix) against the new one. Here the 'old' one is the new one with the added fields
    stripped and the old labels put back, which is what the fix must reproduce number for number."""
    new = merged["disk"]
    old = json.loads(json.dumps(new))
    for k in ("provenance", "field_notes"):
        old.pop(k)
    old["config"]["version"] = "headline-v2"
    old["residue"]["pooled"]["power_rule"]["protocol"] = 2
    diffs, fixes = compare_numeric(old, new)
    assert diffs == [] and len(fixes) == 2


STORED = (("results/headline-300-v3/summary_v3.pre-fix.json", "results/headline-300-v3/summary_v3.json"),
          ("results/headline-u2-v3/summary_v3.pre-fix.json", "results/headline-u2-v3/summary_v3.json"))


@pytest.mark.parametrize("old_p,new_p", STORED)
def test_stored_summaries_are_numerically_unchanged(old_p, new_p):
    old_f, new_f = ROOT / old_p, ROOT / new_p
    if not (old_f.exists() and new_f.exists()):
        pytest.skip("stored results are not on this machine")
    new = json.loads(new_f.read_text())
    if "provenance" not in new:
        pytest.skip("the stored summary has not been regenerated yet")
    diffs, fixes = compare_numeric(json.loads(old_f.read_text()), new)
    assert diffs == []
    assert all(old_v in (2, "headline-v2") and new_v in (3, "headline-v3") for _, old_v, new_v in fixes)


def main(argv: list[str]) -> int:
    old, new = (json.loads(Path(a).read_text()) for a in argv[:2])
    diffs, fixes = compare_numeric(old, new)
    print(f"corrected labels: {len(fixes)}")
    for p, a, b in fixes:
        print(f"  {p}: {a!r} -> {b!r}")
    print(f"differences: {len(diffs)}")
    for d in diffs[:50]:
        print("  " + d)
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


def test_omission_degeneracy_counts_notes_at_one_and_lists_exceptions():
    from lens.flip.headline.metrics import omission_degeneracy
    rows = [{"case_id": "a", "system": "recent", "omission": 1.0},
            {"case_id": "b", "system": "summary", "omission": 0.95},
            {"case_id": "c", "system": "rag-bm25", "omission": None}]
    d = omission_degeneracy(rows)
    assert (d["n_notes"], d["n_known"], d["n_missing"], d["n_at_one"]) == (3, 2, 1, 1)
    assert d["exceptions"] == [{"case_id": "b", "system": "summary", "value": 0.95}] and d["descriptive_only"]
