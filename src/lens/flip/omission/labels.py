"""Parser for the filled calibration file. Reads the Markdown and the key, never edits the Markdown.

A record line is `- [u037] label: y | unit text`. The answer sits between "label:" and the first "|". y, n and ?
are accepted in either case. An empty answer is unlabelled. Anything else is invalid. Lines that are not record
lines are ignored. A key unit with no line in the file is reported as missing. A unit id that the key does not
know, or that appears twice, is reported as invalid.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_LINE = re.compile(r"^\s*[-*]\s*\[(u\d+)\]\s*label\s*:(.*)$", re.IGNORECASE)
VALID = {"y": "y", "n": "n", "?": "?"}


def read_key(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def parse_text(text: str, key: list[dict[str, Any]]) -> dict[str, Any]:
    """{"labels": rows, "unlabelled": [...], "invalid": [...], "missing": [...], "counts": {...}}"""
    by_label = {k["unit_label"]: k for k in key}
    seen: set[str] = set()
    labels, unlabelled, invalid = [], [], []
    for no, line in enumerate(text.splitlines(), start=1):
        m = _LINE.match(line)
        if not m:
            if re.match(r"^\s*[-*]\s*\[u\d+\]", line, re.IGNORECASE):
                invalid.append({"line": no, "reason": "record line without 'label:'", "text": line.strip()[:80]})
            continue
        uid = m.group(1).lower()
        rest = m.group(2)
        if uid not in by_label:
            invalid.append({"line": no, "unit_label": uid, "reason": "unit id not in the key"})
            continue
        if uid in seen:
            invalid.append({"line": no, "unit_label": uid, "reason": "unit id appears twice"})
            continue
        seen.add(uid)
        if "|" not in rest:
            invalid.append({"line": no, "unit_label": uid, "reason": "the '|' after the answer is missing"})
            continue
        answer = rest.split("|", 1)[0].strip()
        if not answer:
            unlabelled.append({"line": no, "unit_label": uid})
        elif answer.lower() in VALID:
            k = by_label[uid]
            labels.append({"unit_label": uid, "note_label": k["note_label"], "case_id": k["case_id"],
                           "system": k["system"], "unit_id": k["unit_id"], "unit_index": k["unit_index"],
                           "label": VALID[answer.lower()]})
        else:
            invalid.append({"line": no, "unit_label": uid, "reason": f"answer {answer[:20]!r} is not y, n or ?"})
    missing = sorted(set(by_label) - seen)
    counts = {"key_units": len(key), "labelled": len(labels), "unlabelled": len(unlabelled),
              "invalid": len(invalid), "missing": len(missing),
              "y": sum(r["label"] == "y" for r in labels), "n": sum(r["label"] == "n" for r in labels),
              "?": sum(r["label"] == "?" for r in labels)}
    labels.sort(key=lambda r: r["unit_label"])
    return {"labels": labels, "unlabelled": unlabelled, "invalid": invalid, "missing": missing, "counts": counts}


def parse_file(md: Path, key_path: Path) -> dict[str, Any]:
    return parse_text(Path(md).read_text(encoding="utf-8"), read_key(Path(key_path)))


def write_labels(path: Path, labels: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in labels:
            fh.write(json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")


def read_labels(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
