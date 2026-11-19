"""The `provenance` block of every headline summary.

It answers four questions about a report, from the files on disk and git, with no network and no clock:

- which protocol the report is (the number, the protocol file, and the commit that last touched that file);
- what each source run's own protocol.json says (version, status, frozen_at, and the commit of its protocol file);
- for a merge, which run is the base and which replaces a mechanism;
- the sha256 of every input file the report read.

The block is deterministic: the same files and the same git state give the same bytes.
`state/cases/*.json` (one file per case, the stored results) is summarised as a count and one aggregate sha256
over the sorted file names and their file hashes.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]

# Files of a run dir that a report reads (or that the report's own row files hold). Missing files are skipped.
RUN_FILES = ("config.json", "cases.jsonl", "protocol.json", "spend.json", "decisions.jsonl", "records.jsonl",
             "residue.jsonl", "residue_records.jsonl")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rel(p: Path | str) -> str:
    """Path relative to the repo root when the file is inside it, else the resolved path."""
    q = Path(p).resolve()
    try:
        return str(q.relative_to(ROOT))
    except ValueError:
        return str(q)


def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def protocol_file(version: int) -> Path:
    return ROOT / "docs" / f"flip-protocol-v{version}.md"


def protocol_file_info(version: int) -> dict[str, Any]:
    """The protocol file of a version: path, sha256, last commit that touched it, and whether it is unmodified."""
    p = protocol_file(version)
    info: dict[str, Any] = {"file": rel(p), "file_sha256": sha256_file(p) if p.exists() else None,
                            "frozen_commit": None, "file_matches_commit": None}
    if p.exists():
        commit = _git("log", "-1", "--format=%H", "--", str(p.relative_to(ROOT))) or None
        info["frozen_commit"] = commit
        if commit:
            info["file_matches_commit"] = _git("status", "--porcelain", "--", str(p.relative_to(ROOT))) == ""
    return info


def run_inputs(run_dir: Path) -> dict[str, Any]:
    files = {n: sha256_file(run_dir / n) for n in RUN_FILES if (run_dir / n).exists()}
    state = sorted((run_dir / "state" / "cases").glob("*.json"))
    agg = hashlib.sha256()
    for p in state:
        agg.update(f"{p.name}\n{sha256_file(p)}\n".encode("utf-8"))
    return {"files": files, "state_cases": {"n": len(state), "sha256": agg.hexdigest() if state else None}}


def run_protocol(run_dir: Path) -> dict[str, Any]:
    """What the run dir's protocol.json says, and where its protocol file stands in git."""
    pj = run_dir / "protocol.json"
    if not pj.exists():
        return {"protocol_json": None}
    rec = json.loads(pj.read_text())
    out: dict[str, Any] = {"protocol_json": {"version": rec.get("version"), "status": rec.get("status"),
                                             "frozen_at": rec.get("frozen_at"), "sha256": sha256_file(pj)}}
    if isinstance(rec.get("version"), int):
        out["protocol_json"].update(protocol_file_info(rec["version"]))
    return out


def build(protocol: int, runs: list[tuple[str, Path]], merge: dict[str, Any] | None = None) -> dict[str, Any]:
    """runs: (role, run dir) pairs, role in run | base | replacement."""
    info = protocol_file_info(protocol)
    out: dict[str, Any] = {
        "protocol": protocol,
        "protocol_file": info["file"],
        "protocol_file_sha256": info["file_sha256"],
        "protocol_frozen_commit": info["frozen_commit"],
        "protocol_file_matches_commit": info["file_matches_commit"],
        "runs": [],
        "inputs_sha256": {},
    }
    for role, d in runs:
        cfg_p = d / "config.json"
        run_cfg = json.loads(cfg_p.read_text()) if cfg_p.exists() else {}
        out["runs"].append({"role": role, "dir": rel(d), "config_version": run_cfg.get("version"), **run_protocol(d)})
        out["inputs_sha256"][rel(d)] = run_inputs(d)
    if merge:
        out["merge"] = merge
    return out
