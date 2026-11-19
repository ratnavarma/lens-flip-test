"""C4 judge v2 (exploratory, outside protocol v5's frozen arms). Local first: Ollama only, no API call.

The v5 C4 judge (the omission-judge-v1 prompt on Haiku) marked 93 in 100 units absent and its semantic-omission
fraction had AUROC 0.50 for the v3 residue flips. This module rebuilds the judge and reruns the C4 analysis on the
same 11,240 (note, unit) pairs. Nothing under results/exploratory-v5/{X1,X2,C1,C2,C3,C4} is written; everything goes
to results/exploratory-v5/judge2/.

    uv run python -m lens.flip.explore5.judge2 calibrate   build the truth set, score every judge, pick one
    uv run python -m lens.flip.explore5.judge2 run         apply the picked judge to the 11,240 pairs, write the summary
    uv run python -m lens.flip.explore5.judge2 report      print the table

Truth. Varma's hand labels in the vault sheet when at least 150 are filled. Otherwise a synthetic set built from the
frozen v3 notes and their view units (recipe in `RECIPE`), six kinds:

    natural_present        the real note carries at least half of the unit's distinguishing numbers (truth yes)
    inserted_verbatim      a strictly absent unit, its text pasted into a copy of the note (yes)
    inserted_paraphrase    the same with a rule-based light paraphrase (dates reworded, phrases swapped) (yes)
    natural_absent         the real note mentions none of the unit's anchors, picked across the cosine range (no)
    deleted_sentence       a natural_present unit with the note sentences that carry it removed (no)
    same_topic_distractor  a strictly absent unit, with a different unit of the same kind pasted into the note (no)

Judges. (a) nomic-embed-text coverage (max cosine of the unit to a note sentence) with tau chosen on the set by
kappa (`omission.calibrate`); (b) qwen3:30b-a3b and (c) qwen3:8b with prompt `c4-judge-v2`, which asks whether the
unit is stated or clearly implied and needs a quote from the note for a yes; (d) each model only on the embedding's
borderline band (90 percent precision thresholds). Pass bar: balanced accuracy >= 0.85 on the set and an absent rate
on real pairs under 0.75.
"""

from __future__ import annotations

import argparse
import calendar
import concurrent.futures as cf
import hashlib
import json
import math
import random
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from ..backends import OLLAMA_URL, OllamaTransport, Transport
from ..cache import CallCache, make_key
from ..headline import requests as rq
from ..omission import calibrate as cal
from ..omission.coverage import coverage, salient_tokens
from ..omission.frame import Frame as OFrame
from ..omission.frame import NoteRec, UnitRec
from ..systems.base import CHARS_PER_TOKEN, split_sentences
from .frame import OUT

OUT2_NAME = "judge2"
PROMPT_VERSION = "c4-judge-v2"
MODELS = {"qwen30": "qwen3:30b-a3b", "qwen8": "qwen3:8b"}
EMBED_MODEL = "nomic-embed-text"
NUM_PREDICT = 160
NUM_CTX = 4096
WORKERS = 4
JUDGE_FORMAT: dict[str, Any] = {
    "type": "object",
    "properties": {"quote": {"type": "string"}, "conveyed": {"type": "boolean"}},
    "required": ["quote", "conveyed"],
}
PASS_BALANCED_ACCURACY = 0.85
PASS_ABSENT_RATE = 0.75          # on real pairs, strictly below
MAX_FULL_HOURS = 4.0
SEED = "x5-judge2"
REAL_SAMPLE = 300                # uniform real pairs: absent rate and the seconds per pair
HAND_LABELS_MIN = 150
VAULT = Path("results/omission-explore-v1/omission-calibration.md")
KEY = Path("results/omission-explore-v1/calibration_key.jsonl")
HAIKU_OUT_TOKENS_V2 = 40         # a short quote plus the boolean

KINDS_YES = ("natural_present", "inserted_verbatim", "inserted_paraphrase")
KINDS_NO = ("natural_absent", "deleted_sentence", "same_topic_distractor")
QUOTA = {"natural_present": 20, "inserted_verbatim": 15, "inserted_paraphrase": 15,
         "natural_absent": 20, "deleted_sentence": 15, "same_topic_distractor": 15}
PER_NOTE_CAP = 3
MAX_INSERT_CHARS = 260
MAX_PRESENT_CHARS = 200
MIN_DELETED_CHARS = 100          # what is left of a note after deleted_sentence
SYSTEMS = ("recent", "rag-bm25", "rag-embed", "summary")

RECIPE = {
    "version": "judge2-calibration-v2",
    "seed": SEED,
    "per_system_quota": QUOTA,
    "per_note_cap": PER_NOTE_CAP,
    "kinds": {
        "natural_present": "real note; the unit has distinguishing numbers (decimal or 3+ digits, dates excluded) and "
                           "at least half of them appear in note sentences about the same kind of record (a payment "
                           "amount in a sentence about a statement does not count); unit body at most 200 chars; "
                           "truth yes",
        "inserted_verbatim": "unit strictly absent from the real note; its body (event id dropped) inserted as a "
                             "sentence at a random sentence boundary of a copy of the note; truth yes",
        "inserted_paraphrase": "as inserted_verbatim with a rule-based light paraphrase (dates as month names, "
                               "phrase table, zero fields as 'no ...'); numbers and ids kept; truth yes",
        "natural_absent": "real note; none of the unit's anchors (numbers, ids, dates in any form, month name with "
                          "the year) appear; picked spread over the cosine range; truth no",
        "deleted_sentence": "a natural_present unit; the note sentences that carry any of its anchors are removed; "
                            "the rest must keep at least 100 chars and be strictly free of its anchors; truth no",
        "same_topic_distractor": "unit strictly absent; a different unit of the same kind (statement, reading, "
                                 "payment, credit, device, contact, notice) from the same view inserted verbatim or "
                                 "paraphrased; the target must stay strictly absent; truth no",
    },
    "strata": "each record carries the omission-explore-v1 stratum of its real pair (verbatim if a salient number "
              "overlaps, else paraphrased or absent by the per-system median cosine) and its real cosine",
    "inserted_max_chars": MAX_INSERT_CHARS,
}


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _jsonl(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Anchors: what would show that a note carries a unit
# ---------------------------------------------------------------------------

_LEAD = re.compile(r"^e\d{3,}\s+")
_LEAD_DATE = re.compile(r"^(\d{4}-\d{2}-\d{2})\s+")
_DATE = re.compile(r"(?<![\d-])(\d{4})-(\d{2})(?:-(\d{2}))?(?![\d])")
_ID = re.compile(r"\b(cr\d{3,}|CH-\d{4}-\d{2})\b", re.IGNORECASE)
_CITE = re.compile(r"\s*\((?:e\d{3,}(?:,\s*)?)+\)")


def body(unit: str) -> str:
    """The unit without its leading event id."""
    return _LEAD.sub("", unit.strip())


def distinguishing_numbers(text: str) -> set[str]:
    """Salient numbers that are not dates (decimal point or at least 3 digits, not zero)."""
    return {t for t in salient_tokens(text) if "-" not in t}


def unit_dates(text: str) -> list[tuple[int, int, int | None]]:
    out = []
    for m in _DATE.finditer(body(text)):
        y, mo = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12:
            out.append((y, mo, int(m.group(3)) if m.group(3) else None))
    return out


def date_forms(y: int, mo: int, d: int | None) -> set[str]:
    """Lower-case strings that would mention this date (or its month) in a note."""
    ab, full = calendar.month_abbr[mo].lower(), calendar.month_name[mo].lower()
    f = {f"{y}-{mo:02d}", f"{ab} {y}", f"{full} {y}", f"{ab}. {y}", f"{ab}-{y}", f"{mo:02d}/{y}", f"{y}/{mo:02d}",
         f"{ab} '{y % 100:02d}", f"{ab}{y % 100:02d}"}
    if d is not None:
        f |= {f"{mo:02d}-{d:02d}", f"{ab} {d}", f"{full} {d}", f"{mo}/{d}", f"{d} {ab}"}
    return f


def mentions(note: str, unit: str) -> bool:
    """True when the note mentions any anchor of the unit: a distinguishing number, a credit or charge id, a date or
    month in any common form, or the month's name together with the year anywhere in the note."""
    low = note.lower()
    if distinguishing_numbers(unit) & distinguishing_numbers(note):
        return True
    if any(i.lower() in low for i in _ID.findall(body(unit))):
        return True
    for y, mo, d in unit_dates(unit):
        for f in date_forms(y, mo, d):
            if re.search(r"(?<![\w])" + re.escape(f) + r"(?![\d])", low):
                return True
        names = (calendar.month_abbr[mo].lower(), calendar.month_name[mo].lower())
        if str(y) in low and any(re.search(rf"\b{n}\b", low) for n in names):
            return True
    return False


def strictly_absent(note: str, unit: str) -> bool:
    """Truth no for a natural pair: the unit has a date, and the note mentions none of its anchors."""
    return bool(unit_dates(unit) or distinguishing_numbers(unit) or _ID.search(body(unit))) and not mentions(note, unit)


# A matched number counts only in a note sentence about the same kind of record: a payment of 138.42 is not
# conveyed by "statement (138.42)", nor a credit by the one-off charge it offsets.
_KIND_WORDS = {
    "statement": r"statement|stmt|charge|due|total|activity",
    "reading": r"reading|read|units|actual|estimat|usage",
    "payment": r"paid|payment|pay\b|posted",
    "credit": r"credit|\bcr\d",
}


def naturally_present(note: str, unit: str) -> bool:
    """Truth yes for a natural pair: at least half (and at least one) of the unit's distinguishing numbers are in
    note sentences about the same kind of record, and the unit is short enough to be one fact."""
    nums = distinguishing_numbers(unit)
    if not nums or len(body(unit)) > MAX_PRESENT_CHARS:
        return False
    words = _KIND_WORDS.get(kind_of(unit))
    found: set[str] = set()
    for sent in split_sentences(note):
        if words is None or re.search(words, sent.lower()):
            found |= nums & distinguishing_numbers(sent)
    return len(found) >= max(1, math.ceil(len(nums) / 2))


def mechanical_class(note: str, unit: str) -> str:
    """present_rule, strict_absent or ambiguous: the rule-based truth classes of a real pair."""
    if naturally_present(note, unit):
        return "present_rule"
    return "strict_absent" if strictly_absent(note, unit) else "ambiguous"


def kind_of(unit: str) -> str:
    b = body(unit).lower()
    for k, pat in (("contact", r"reached out|contact|complaint|by (phone|chat|email|mail)"),
                   ("statement", r"statement|charge[sd]? \d"), ("reading", r"reading|read\b"),
                   ("payment", r"payment|paid"), ("credit", r"credit"), ("device", r"device|equipment|technician"),
                   ("notice", r"notice|plan")):
        if re.search(pat, b):
            return k
    return "other"


# ---------------------------------------------------------------------------
# Light paraphrase, insertion, deletion
# ---------------------------------------------------------------------------

_PHRASES: tuple[tuple[str, str], ...] = (
    (r",?\s*other charges 0\.00", ", no other charges"),
    (r",?\s*credits 0\.00", ", no credits"),
    (r"^Account statement\b", "The invoice for"),
    (r"^Statement for period\b", "The invoice for"),
    (r"^Statement\b", "The invoice for"),
    (r"\bstatement\b", "invoice"),
    (r"\bactivity charges\b", "usage charges"),
    (r"\bTechnician serviced the device\b", "A technician did service work on the equipment"),
    (r"\bRoutine service visit on the device\b", "The equipment had a routine service visit"),
    (r"\bA new device was installed at the account\b", "The account got new equipment"),
    (r"^Device replaced\b", "The equipment was swapped out"),
    (r"^Device serviced\b", "The equipment was serviced"),
    (r"^Applied credit (\S+?)\.?$", r"Credit \1 was applied."),
    (r"\bThe customer reached out by (\w+) about:?", r"The customer got in touch by \1 about"),
    (r"\bReading taken on site for\b", "On-site read for"),
    (r"\bEstimated reading for\b", "Estimate for"),
    (r"\bActual reading for\b", "Actual read for"),
    (r"\bNotice sent\.", "A notice went out:"),
    (r"\bwas posted against the balance\b", "went onto the balance"),
    (r"\bdevice\b", "equipment"),
)


def _human_date(m: re.Match) -> str:
    y, mo, d = int(m.group(1)), int(m.group(2)), m.group(3)
    if not 1 <= mo <= 12:
        return m.group(0)
    if d:
        return f"{calendar.month_abbr[mo]} {int(d)}, {y}"
    return f"{calendar.month_name[mo]} {y}"


def paraphrase(unit: str) -> str:
    """A rule-based light paraphrase: event id and citations dropped, phrase table applied, dates as month names.
    Every distinguishing number and id of the unit is kept (test)."""
    b = body(unit)
    lead = _LEAD_DATE.match(b)
    when = None
    if lead:
        when = lead.group(1)
        b = b[lead.end():]
    b = _CITE.sub("", b)
    for pat, rep in _PHRASES:
        b = re.sub(pat, rep, b)
    b = b.strip()
    if when:
        b = b.rstrip(".") + f" (dated {when})."
    b = _DATE.sub(_human_date, b)
    b = re.sub(r"\s+", " ", b).strip()
    if b and not b.endswith((".", "!", "?")):
        b += "."
    return b[:1].upper() + b[1:]


def verbatim_sentence(unit: str) -> str:
    b = body(unit).strip()
    return b if b.endswith((".", "!", "?")) else b + "."


def insert_sentence(note: str, sentence: str, rng: random.Random) -> str:
    """The note with one sentence inserted at a random sentence boundary (never after the last, cut-off piece)."""
    ss = split_sentences(note)
    pos = rng.randrange(0, len(ss)) if ss else 0
    return " ".join(ss[:pos] + [sentence] + ss[pos:])


def delete_carrying(note: str, unit: str) -> str | None:
    """The note without every sentence that mentions an anchor of the unit, or None if nothing usable is left."""
    ss = split_sentences(note)
    keep = [s for s in ss if not mentions(s, unit)]
    if not keep or len(keep) == len(ss):
        return None
    out = " ".join(keep)
    return out if len(out) >= MIN_DELETED_CHARS and strictly_absent(out, unit) else None


# ---------------------------------------------------------------------------
# Embedding coverage on arbitrary pairs (omission.coverage on an ad hoc frame)
# ---------------------------------------------------------------------------

def pair_cosines(pairs: Sequence[tuple[str, str]], embedder: Callable) -> list[float]:
    """Max cosine of each unit to the sentences of its note. Pairs sharing a note text share one NoteRec."""
    groups: dict[str, list[int]] = {}
    for i, (n, _) in enumerate(pairs):
        groups.setdefault(n, []).append(i)
    notes = []
    for n, idx in groups.items():
        notes.append(NoteRec("x", "x", n, None, False, None,
                             [UnitRec(k, f"u{k}", pairs[i][1], False) for k, i in enumerate(idx)]))
    fr = OFrame(notes=notes, case_ids=[], bootstrap={}, power={})
    cos = coverage(fr, embedder)
    out = [0.0] * len(pairs)
    for (n, idx), row in zip(groups.items(), cos):
        for i, c in zip(idx, row):
            out[i] = float(c)
    return out


# ---------------------------------------------------------------------------
# The v2 prompt, parser and local judge
# ---------------------------------------------------------------------------

def build_prompt(note: str, unit: str) -> str:
    return (
        "You check one record against a handoff note. The note was written from many records and is short, "
        "so most records are not in it.\n\n"
        f"Note:\n<<<\n{note.strip()}\n>>>\n\n"
        f"Record:\n<<<\n{unit.strip()}\n>>>\n\n"
        "Is the content of this record stated in the note, or clearly implied by it?\n"
        "- Yes when the note gives the same fact in any wording. Other date formats, rounding and short forms count.\n"
        "- Yes when the note clearly covers this record inside a group, such as \"all 2016 statements were paid\" "
        "for a 2016 payment.\n"
        "- No when the note only has a similar record with another date, period or amount.\n"
        "- No when the note only shares the topic or a few words.\n"
        "- The record id at the start (like e0123) does not need to appear in the note.\n\n"
        "Reply in JSON with two fields.\n"
        "\"quote\": if yes, copy the words from the NOTE (not from the record) that state or imply the record, at "
        "most 15 words. If no, an empty string.\n"
        "\"conveyed\": true or false."
    )


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_QCHARS = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-",
                         "…": "..."})


def _norm(s: str) -> str:
    s = s.translate(_QCHARS).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip(" .,;:'\"()[]")


def quote_in_note(quote: str, note: str) -> bool:
    """The quote (or each piece of it between ellipses) appears in the note, ignoring case, spacing and edge
    punctuation. Pieces shorter than 3 characters are ignored; at least one piece must be checked."""
    n = _norm(note)
    pieces = [_norm(p) for p in re.split(r"\.\.\.|…", quote.translate(_QCHARS))]
    pieces = [p for p in pieces if len(p) >= 3]
    return bool(pieces) and all(p in n for p in pieces)


def parse_reply(raw: str | None, note: str) -> dict[str, Any]:
    """{"conveyed": bool | None, "raw_conveyed", "quote", "status"}. status: ok, unsupported (a yes whose quote is
    not in the note: counted as no), parse (not JSON in the schema: None)."""
    if raw is None:
        return {"conveyed": None, "raw_conveyed": None, "quote": None, "status": "parse"}
    try:
        obj = json.loads(_THINK.sub("", raw).strip())
    except ValueError:
        return {"conveyed": None, "raw_conveyed": None, "quote": None, "status": "parse"}
    if not (isinstance(obj, dict) and isinstance(obj.get("conveyed"), bool)):
        return {"conveyed": None, "raw_conveyed": None, "quote": None, "status": "parse"}
    q = obj.get("quote") if isinstance(obj.get("quote"), str) else ""
    if obj["conveyed"] and not quote_in_note(q, note):
        return {"conveyed": False, "raw_conveyed": True, "quote": q, "status": "unsupported"}
    return {"conveyed": obj["conveyed"], "raw_conveyed": obj["conveyed"], "quote": q, "status": "ok"}


class OllamaJudge2:
    """temperature 0, seed 0, think off, the two-field schema. Cached by request (CallCache, main thread only);
    the misses go to Ollama from a small thread pool."""

    def __init__(self, model: str, *, transport: Transport | None = None, cache: CallCache | None = None,
                 url: str = OLLAMA_URL, workers: int = WORKERS):
        self.model = model
        self.transport = transport or OllamaTransport(url)
        self.cache = cache
        self.workers = max(1, workers)
        self.real_calls = 0
        self.cache_hits = 0
        self.seconds = 0.0

    def _req(self, note: str, unit: str) -> tuple[str, dict]:
        messages = [{"role": "user", "content": build_prompt(note, unit)}]
        options = {"temperature": 0, "seed": 0, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT}
        key = make_key(self.model, PROMPT_VERSION, messages, options, {"think": False, "format": JUDGE_FORMAT})
        body_ = {"model": self.model, "messages": messages, "stream": False, "think": False,
                 "format": JUDGE_FORMAT, "options": options}
        return key, body_

    def many(self, pairs: Sequence[tuple[str, str]], log: Callable[[str], None] | None = None,
             every: int = 250) -> list[dict[str, Any]]:
        raws: list[str | None] = [None] * len(pairs)
        todo: list[tuple[int, str, dict]] = []
        for i, (n, u) in enumerate(pairs):
            key, b = self._req(n, u)
            hit = self.cache.get(key) if self.cache is not None else None
            if hit is not None:
                raws[i] = hit
                self.cache_hits += 1
            else:
                todo.append((i, key, b))
        t0 = time.monotonic()
        done = 0
        if todo:
            with cf.ThreadPoolExecutor(self.workers) as ex:
                futs = {ex.submit(self.transport, b): (i, key) for i, key, b in todo}
                for f in cf.as_completed(futs):
                    i, key = futs[f]
                    raw = f.result()
                    raws[i] = raw
                    self.real_calls += 1
                    if self.cache is not None:
                        self.cache.put(key, self.model, PROMPT_VERSION, raw)
                    done += 1
                    if log and done % every == 0:
                        dt = time.monotonic() - t0
                        log(f"  {self.model}: {done}/{len(todo)} new calls, {dt / done:.2f} s/pair")
        self.seconds += time.monotonic() - t0
        return [parse_reply(r, n) for r, (n, _) in zip(raws, pairs)]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def score(pred: Sequence[bool | None], truth: Sequence[bool]) -> dict[str, Any]:
    """Balanced accuracy, kappa and the confusion counts. A parse failure (None) counts as wrong."""
    fixed = [(not t) if p is None else bool(p) for p, t in zip(pred, truth)]
    m = cal.metrics(fixed, list(truth))
    rec, spec = m["recall"], m["specificity"]
    ba = None if rec is None or spec is None else (rec + spec) / 2
    return {"n": m["n"], "balanced_accuracy": ba, "kappa": m["kappa"], "accuracy": m["agreement"],
            "recall": rec, "specificity": spec, "precision": m["precision"],
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
            "parse_failures": sum(p is None for p in pred)}


def absent_rate(pred: Sequence[bool | None]) -> dict[str, Any]:
    ok = [p for p in pred if p is not None]
    return {"value": (sum(not p for p in ok) / len(ok)) if ok else None, "n": len(ok),
            "unparsed": len(pred) - len(ok)}


def hybrid(cos: Sequence[float], judged: Sequence[bool | None], lo: float | None, hi: float | None
           ) -> list[bool | None]:
    """Embedding outside the band (>= hi conveyed, <= lo absent), the judge inside it."""
    out: list[bool | None] = []
    for c, j in zip(cos, judged):
        if hi is not None and c >= hi:
            out.append(True)
        elif lo is not None and c <= lo:
            out.append(False)
        else:
            out.append(j)
    return out


def passes(ba: float | None, ar: float | None) -> bool:
    return ba is not None and ar is not None and ba >= PASS_BALANCED_ACCURACY and ar < PASS_ABSENT_RATE


# ---------------------------------------------------------------------------
# Real pairs and the calibration set
# ---------------------------------------------------------------------------

def real_pairs(handoffs) -> list[dict[str, Any]]:
    """The C4 pairs in engine order: every unit of every view against the v3 note."""
    out = []
    for h in handoffs:
        note = h.row["v3"]["note"]
        for j, u in enumerate(h.view.units):
            out.append({"case_id": h.case.case_id, "system": h.system, "unit_index": j, "note": note, "unit": u})
    return out


def strata(rp: Sequence[dict], cos: Sequence[float]) -> list[str]:
    """The omission-explore-v1 strata of the real pairs: verbatim if a salient number overlaps, else paraphrased or
    absent by the per-system median cosine of the non-overlap pairs."""
    from ..omission.coverage import number_overlap
    ov = [number_overlap(p["unit"], p["note"]) for p in rp]
    cut = {}
    for s in {p["system"] for p in rp}:
        rest = [c for p, c, o in zip(rp, cos, ov) if p["system"] == s and not o]
        cut[s] = statistics.median(rest) if rest else 0.0
    return ["verbatim" if o else ("paraphrased" if c >= cut[p["system"]] else "absent")
            for p, c, o in zip(rp, cos, ov)]


def _spread(cands: list[dict], q: int, rng: random.Random) -> list[dict]:
    if q <= 0 or not cands:
        return []
    ordered = sorted(cands, key=lambda r: (r["real_cosine"], r["case_id"], r["unit_index"]))
    n, q = len(ordered), min(q, len(ordered))
    return [ordered[rng.randrange((b * n) // q, ((b + 1) * n) // q)] for b in range(q)]


def build_calibration_set(rp: Sequence[dict], cos: Sequence[float], seed: str = SEED,
                          quota: dict[str, int] | None = None) -> list[dict[str, Any]]:
    """Deterministic synthetic truth set. `rp` are the real pairs, `cos` their cosines (same order)."""
    quota = dict(quota or QUOTA)
    st = strata(rp, cos)
    rows: list[dict[str, Any]] = []
    for si, system in enumerate(SYSTEMS):
        rng = random.Random(f"{seed}|{system}")
        notes: dict[tuple[str, str], list[int]] = {}
        for i, p in enumerate(rp):
            if p["system"] == system:
                notes.setdefault((p["case_id"], p["system"]), []).append(i)
        keys = sorted(notes)
        rng.shuffle(keys)
        used: dict[tuple[str, str], int] = {}
        taken: set[int] = set()

        def base(i: int, kind: str) -> dict[str, Any]:
            p = rp[i]
            return {"kind": kind, "truth": kind in KINDS_YES, "case_id": p["case_id"], "system": system,
                    "unit_index": p["unit_index"], "unit": p["unit"], "real_note": p["note"],
                    "real_cosine": round(float(cos[i]), 6), "real_stratum": st[i], "unit_kind": kind_of(p["unit"])}

        def room(k: tuple[str, str]) -> bool:
            return used.get(k, 0) < PER_NOTE_CAP

        def take(k: tuple[str, str], i: int, rec: dict) -> None:
            used[k] = used.get(k, 0) + 1
            taken.add(i)
            rows.append(rec)

        present = [i for k in keys for i in notes[k] if naturally_present(rp[i]["note"], rp[i]["unit"])]
        absent = [i for k in keys for i in notes[k] if strictly_absent(rp[i]["note"], rp[i]["unit"])]

        # natural_present and natural_absent: spread over the cosine range, one note at most PER_NOTE_CAP
        for kind, pool in (("natural_present", present), ("natural_absent", absent)):
            cands = [{"i": i, "real_cosine": cos[i], "case_id": rp[i]["case_id"], "unit_index": rp[i]["unit_index"]}
                     for i in pool]
            picks = _spread(cands, quota[kind] * 2, rng)
            got = 0
            for c in picks:
                k = (rp[c["i"]]["case_id"], system)
                if got >= quota[kind] or c["i"] in taken or not room(k):
                    continue
                r = base(c["i"], kind)
                r.update({"note": rp[c["i"]]["note"], "note_modified": False})
                take(k, c["i"], r)
                got += 1

        # deleted_sentence: a present unit, its carrying sentences removed
        got = 0
        for i in rng.sample(present, len(present)):
            if got >= quota["deleted_sentence"]:
                break
            k = (rp[i]["case_id"], system)
            if i in taken or not room(k):
                continue
            nn = delete_carrying(rp[i]["note"], rp[i]["unit"])
            if nn is None:
                continue
            r = base(i, "deleted_sentence")
            r.update({"note": nn, "note_modified": True})
            take(k, i, r)
            got += 1

        # inserted_verbatim, inserted_paraphrase: an absent unit pasted into a copy of the note
        for kind, fn in (("inserted_verbatim", verbatim_sentence), ("inserted_paraphrase", paraphrase)):
            got = 0
            for i in rng.sample(absent, len(absent)):
                if got >= quota[kind]:
                    break
                k = (rp[i]["case_id"], system)
                if i in taken or not room(k) or len(body(rp[i]["unit"])) > MAX_INSERT_CHARS:
                    continue
                s = fn(rp[i]["unit"])
                r = base(i, kind)
                r.update({"note": insert_sentence(rp[i]["note"], s, rng), "note_modified": True, "inserted": s})
                take(k, i, r)
                got += 1

        # same_topic_distractor: an absent unit; a different unit of the same kind pasted into the note
        got = 0
        for i in rng.sample(absent, len(absent)):
            if got >= quota["same_topic_distractor"]:
                break
            k = (rp[i]["case_id"], system)
            if i in taken or not room(k):
                continue
            kd = kind_of(rp[i]["unit"])
            if kd == "other":
                continue
            others = [j for j in notes[k] if j != i and kind_of(rp[j]["unit"]) == kd
                      and body(rp[j]["unit"]) != body(rp[i]["unit"]) and len(body(rp[j]["unit"])) <= MAX_INSERT_CHARS]
            rng.shuffle(others)
            for j in others:
                s = verbatim_sentence(rp[j]["unit"]) if rng.random() < 0.5 else paraphrase(rp[j]["unit"])
                nn = insert_sentence(rp[i]["note"], s, rng)
                if strictly_absent(nn, rp[i]["unit"]):
                    r = base(i, "same_topic_distractor")
                    r.update({"note": nn, "note_modified": True, "inserted": s,
                              "distractor_unit_index": rp[j]["unit_index"]})
                    take(k, i, r)
                    got += 1
                    break
    for n, r in enumerate(rows, start=1):
        r["id"] = f"c{n:03d}"
    return rows


def set_from_hand_labels(rp: Sequence[dict], cos: Sequence[float], labels: Sequence[dict]) -> list[dict[str, Any]]:
    """Truth from Varma's y/n labels ("?" left out). Keys are (case_id, system, unit_index) of the real pairs."""
    at = {(p["case_id"], p["system"], p["unit_index"]): i for i, p in enumerate(rp)}
    st = strata(rp, cos)
    rows = []
    for lab in labels:
        if lab["label"] not in ("y", "n"):
            continue
        i = at[(lab["case_id"], lab["system"], lab["unit_index"])]
        p = rp[i]
        rows.append({"id": lab["unit_label"], "kind": "hand_label", "truth": lab["label"] == "y",
                     "case_id": p["case_id"], "system": p["system"], "unit_index": p["unit_index"], "unit": p["unit"],
                     "note": p["note"], "real_note": p["note"], "note_modified": False,
                     "real_cosine": round(float(cos[i]), 6), "real_stratum": st[i], "unit_kind": kind_of(p["unit"])})
    return rows


def count_hand_labels(vault: Path = VAULT, key: Path = KEY) -> dict[str, Any]:
    from ..omission import labels as L
    if not vault.exists() or not key.exists():
        return {"vault": str(vault), "exists": vault.exists(), "labelled": 0, "y": 0, "n": 0, "?": 0, "labels": []}
    res = L.parse_file(vault, key)
    c = res["counts"]
    return {"vault": str(vault), "exists": True, "labelled": c["labelled"], "y": c["y"], "n": c["n"], "?": c["?"],
            "labels": res["labels"]}


# ---------------------------------------------------------------------------
# Haiku plan (arithmetic only)
# ---------------------------------------------------------------------------

def haiku_plan_v2(pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """Priced Haiku Batch plan for the v2 prompt. No API call, nothing submitted. Input tokens per pair =
    ceil(prompt chars / 2.2) + 331 (the v5 structured-output overhead); output 40 tokens per pair."""
    from .frame import OVERHEAD_TOKENS
    from ..omission.judge import HAIKU_RATES
    tin = [math.ceil(len(build_prompt(n, u)) / CHARS_PER_TOKEN) + OVERHEAD_TOKENS for n, u in pairs]
    usd = sum(rq.request_price(t, HAIKU_OUT_TOKENS_V2, HAIKU_RATES) for t in tin)
    return {"judge": "claude-haiku-5-5 via the Message Batches API, thinking disabled, prompt c4-judge-v2",
            "n_pairs": len(tin), "input_tokens": sum(tin), "output_tokens": HAIKU_OUT_TOKENS_V2 * len(tin),
            "rates_usd_per_million": dict(HAIKU_RATES), "estimated_usd": round(usd, 4), "submitted": False,
            "note": "Estimate only. Needs Varma's yes before any batch is submitted."}


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _out(root: Path) -> Path:
    p = root / OUT2_NAME
    p.mkdir(parents=True, exist_ok=True)
    return p


def _embedder(cache: CallCache, url: str):
    from ..llm import CachedEmbedder, OllamaEmbedTransport
    return CachedEmbedder(EMBED_MODEL, transport=OllamaEmbedTransport(url), cache=cache)


def _real(root: Path, emb) -> tuple[list, list[dict], list[float]]:
    from .frame import load_frame
    hs = load_frame(root)
    rp = real_pairs(hs)
    t0 = time.monotonic()
    cos = pair_cosines([(p["note"], p["unit"]) for p in rp], emb)
    _log(f"coverage on {len(rp)} real pairs in {time.monotonic() - t0:.1f} s")
    return hs, rp, cos


def _haiku_answers(root: Path, rp: Sequence[dict]) -> list[bool | None] | None:
    from ..explore4.analysis import _states
    s = _states(root, "C4")
    if not s:
        return None
    out = []
    for p in rp:
        conv = s[(p["case_id"], p["system"])]["out"]["conveyed"]
        out.append(conv[p["unit_index"]])
    return out


def cmd_calibrate(args: argparse.Namespace) -> int:
    root = Path(args.root)
    out = _out(root)
    cache = CallCache(out / "cache.sqlite")
    emb = _embedder(cache, args.ollama_url)
    t_all = time.monotonic()
    hl = count_hand_labels()
    _log(f"hand labels in the vault sheet: {hl['labelled']} (y {hl['y']}, n {hl['n']}, ? {hl['?']})")
    hs, rp, cos = _real(root, emb)
    _jsonl(out / "coverage.jsonl", [{"case_id": p["case_id"], "system": p["system"], "unit_index": p["unit_index"],
                                     "cosine": round(c, 6)} for p, c in zip(rp, cos)])

    if hl["labelled"] - hl["?"] >= HAND_LABELS_MIN:
        cs = set_from_hand_labels(rp, cos, hl["labels"])
        recipe = {"version": "judge2-hand-labels", "source": hl["vault"], "n": len(cs)}
    else:
        cs = build_calibration_set(rp, cos)
        recipe = dict(RECIPE)
    text = "".join(json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n" for r in cs)
    setp = out / "calibration_set.jsonl"
    if setp.exists() and setp.read_text(encoding="utf-8") != text and not args.rebuild:
        _log(f"refused: {setp} exists and differs from the rebuilt set (use --rebuild)")
        return 1
    setp.write_text(text, encoding="utf-8")
    recipe.update({"n": len(cs), "sha256": _sha(text), "hand_labels_found": hl["labelled"],
                   "hand_labels_min": HAND_LABELS_MIN,
                   "by_kind": {k: sum(r["kind"] == k for r in cs) for k in sorted({r["kind"] for r in cs})},
                   "by_system": {s: sum(r["system"] == s for r in cs) for s in SYSTEMS},
                   "by_real_stratum": {s: sum(r["real_stratum"] == s for r in cs)
                                       for s in ("verbatim", "paraphrased", "absent")},
                   "truth": {"yes": sum(r["truth"] for r in cs), "no": sum(not r["truth"] for r in cs)}})
    (out / "calibration_recipe.json").write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _log(f"calibration set: {len(cs)} pairs, {recipe['by_kind']}")
    truth = [r["truth"] for r in cs]
    pairs = [(r["note"], r["unit"]) for r in cs]
    ccos = pair_cosines(pairs, emb)

    # (a) embedding
    items = list(zip(ccos, truth))
    tk = cal.choose_tau_kappa(items)
    band = cal.choose_band(items)
    tau, lo, hi = tk["tau"], band["tau_low"], band["tau_high"]
    emb_pred = [c >= tau for c in ccos]
    res: dict[str, Any] = {}
    res["embedding"] = {"judge": f"{EMBED_MODEL} cosine >= tau", "tau": tau, "in_sample": True,
                        "calibration": score(emb_pred, truth),
                        "absent_rate_real": absent_rate([c >= tau for c in cos]), "absent_rate_on": "all real pairs",
                        "seconds_per_pair": 0.0}
    nb = sum(cal.in_band(c, lo, hi) for c in cos)
    band_info = {**band, "borderline_real_pairs": nb, "borderline_share": nb / len(cos)}

    # real sample (uniform, seeded): absent rate and rate
    rng = random.Random(f"{SEED}|real-sample")
    samp = sorted(rng.sample(range(len(rp)), REAL_SAMPLE))
    s_pairs = [(rp[i]["note"], rp[i]["unit"]) for i in samp]
    s_cos = [cos[i] for i in samp]
    res["embedding"]["absent_rate_real_sample"] = absent_rate([c >= tau for c in s_cos])

    decisions = [{"id": r["id"], "kind": r["kind"], "truth": r["truth"], "cosine": round(c, 6),
                  "embedding": p} for r, c, p in zip(cs, ccos, emb_pred)]
    sample_rows = [{"case_id": rp[i]["case_id"], "system": rp[i]["system"], "unit_index": rp[i]["unit_index"],
                    "cosine": round(cos[i], 6)} for i in samp]
    models = [m for m in args.models.split(",") if m]
    for tag in models:
        model = MODELS[tag]
        j = OllamaJudge2(model, cache=cache, url=args.ollama_url, workers=args.workers)
        _log(f"{model}: {len(pairs)} calibration pairs")
        got = j.many(pairs, _log)
        t_cal, n_cal = j.seconds, j.real_calls
        _log(f"{model}: {REAL_SAMPLE} real pairs (rate and absent rate)")
        sg = j.many(s_pairs, _log)
        t_s, n_s = j.seconds - t_cal, j.real_calls - n_cal
        ratep = out / "rates.json"
        rates = json.loads(ratep.read_text()) if ratep.exists() else {}
        if n_s >= 100:                     # a real measurement on uncached real pairs replaces the stored one
            rates[model] = {"seconds_per_pair": t_s / n_s, "n_calls": n_s, "workers": args.workers,
                            "measured_on": "uncached uniform real pairs", "at": time.strftime("%Y-%m-%d %H:%M")}
            ratep.write_text(json.dumps(rates, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        sec = rates.get(model, {}).get("seconds_per_pair")
        if sec is None and n_cal:
            sec = t_cal / n_cal
        pred = [g["conveyed"] for g in got]
        sp = [g["conveyed"] for g in sg]
        res[tag] = {"judge": f"{model} prompt {PROMPT_VERSION}", "calibration": score(pred, truth),
                    "absent_rate_real": absent_rate(sp), "absent_rate_on": f"{REAL_SAMPLE} uniform real pairs",
                    "unsupported_yes": {"calibration": sum(g["status"] == "unsupported" for g in got),
                                        "real_sample": sum(g["status"] == "unsupported" for g in sg)},
                    "raw_yes_scored": score([g["raw_conveyed"] for g in got], truth),
                    "seconds_per_pair": sec, "rate": rates.get(model),
                    "full_run_hours": None if sec is None else sec * len(rp) / 3600}
        hp = hybrid(ccos, pred, lo, hi)
        hsp = hybrid(s_cos, sp, lo, hi)
        res[f"{tag}_band"] = {"judge": f"embedding outside [{lo}, {hi}], {model} inside",
                              "calibration": score(hp, truth), "absent_rate_real": absent_rate(hsp),
                              "absent_rate_on": f"{REAL_SAMPLE} uniform real pairs",
                              "judged_in_band_calibration": sum(cal.in_band(c, lo, hi) for c in ccos),
                              "seconds_per_pair": sec,
                              "full_run_hours": None if sec is None else sec * nb / 3600}
        for d, g, h in zip(decisions, got, hp):
            d[tag] = g["conveyed"]
            d[f"{tag}_status"] = g["status"]
            d[f"{tag}_quote"] = g["quote"]
            d[f"{tag}_band"] = h
        for row, g in zip(sample_rows, sg):
            row[tag] = g["conveyed"]
            row[f"{tag}_status"] = g["status"]

    # by kind and stratum for every judge
    names = [k for k in res]
    for name in names:
        key = "embedding" if name == "embedding" else name
        pred = [d[key] for d in decisions]
        res[name]["by_kind"] = {k: score([p for p, r in zip(pred, cs) if r["kind"] == k],
                                         [r["truth"] for r in cs if r["kind"] == k])["accuracy"]
                                for k in recipe["by_kind"]}
        res[name]["by_real_stratum"] = {s: score([p for p, r in zip(pred, cs) if r["real_stratum"] == s],
                                                 [r["truth"] for r in cs if r["real_stratum"] == s])["accuracy"]
                                        for s in ("verbatim", "paraphrased", "absent")}
        c = res[name]["calibration"]
        res[name]["passes"] = passes(c["balanced_accuracy"], res[name]["absent_rate_real"]["value"])

    # the v5 Haiku answers on the unmodified (real) pairs of the set, for reference (no API call)
    ha = _haiku_answers(root, rp)
    if ha is not None:
        at = {(p["case_id"], p["system"], p["unit_index"]): i for i, p in enumerate(rp)}
        nat = [r for r in cs if not r["note_modified"]]
        hp = [ha[at[(r["case_id"], r["system"], r["unit_index"])]] for r in nat]
        res["haiku_v1"] = {"judge": "claude-haiku-5-5, omission-judge-v1 prompt (the v5 C4 run)",
                           "calibration": score(hp, [r["truth"] for r in nat]),
                           "calibration_subset": "unmodified pairs only (natural_present, natural_absent)",
                           "absent_rate_real": absent_rate(ha), "absent_rate_on": "all real pairs",
                           "passes": None}
        for name in names:                                  # the same subset for every judge, for a fair read
            key = "embedding" if name == "embedding" else name
            idx = {d["id"]: d for d in decisions}
            res[name]["natural_subset"] = score([idx[r["id"]][key] for r in nat], [r["truth"] for r in nat])

    # base rate: the mechanical classes of the real pairs, and what each judge says per class on the real sample
    mclass = [mechanical_class(p["note"], p["unit"]) for p in rp]
    base_rate: dict[str, Any] = {
        "classes": {"present_rule": "naturally_present", "strict_absent": "strictly_absent",
                    "ambiguous": "neither (an anchor of the unit appears, but not its numbers)"},
        "all_real_pairs": {c: {"n": mclass.count(c), "share": mclass.count(c) / len(rp)}
                           for c in ("present_rule", "strict_absent", "ambiguous")},
        "absent_rate_floor_if_every_ambiguous_pair_is_conveyed": mclass.count("strict_absent") / len(rp),
        "real_sample_by_class": {}}
    for c in ("present_rule", "strict_absent", "ambiguous"):
        rows_c = [row for row, i in zip(sample_rows, samp) if mclass[i] == c]
        base_rate["real_sample_by_class"][c] = {"n": len(rows_c), **{
            tag: {"conveyed": sum(bool(r.get(tag)) for r in rows_c)} for tag in models}}

    # pick
    cands = [(n, r) for n, r in res.items() if r.get("passes")]
    choice: dict[str, Any]
    if not cands:
        border = [(rp[i]["note"], rp[i]["unit"]) for i, c in enumerate(cos) if cal.in_band(c, lo, hi)]
        cal_plan = haiku_plan_v2(pairs + s_pairs)
        all_plan = haiku_plan_v2([(p["note"], p["unit"]) for p in rp])
        bl_plan = haiku_plan_v2(border)
        choice = {"status": "no_local_judge_passed", "haiku_plan_calibration": cal_plan,
                  "haiku_plan_all_pairs": all_plan, "haiku_plan_borderline": bl_plan}
        plan = {"status": "plan only: nothing submitted, no API call, no count_tokens call",
                "why": "no local judge passed both bars (balanced accuracy >= 0.85 and real absent rate < 0.75)",
                "steps": [
                    {"step": 1, "what": f"Haiku judge v2 on the {len(pairs)} calibration pairs and the "
                                        f"{len(s_pairs)} real sample pairs; the same pass bar", "plan": cal_plan},
                    {"step": 2, "what": "if it passes: all 11,240 real pairs", "plan": all_plan},
                    {"step": "2 (cheaper)", "what": f"or only the {len(border)} pairs of the embedding band "
                                                    f"[{lo}, {hi}]", "plan": bl_plan}],
                "likely_total_usd": {"calibration_plus_all_pairs": round(cal_plan["estimated_usd"] + all_plan["estimated_usd"], 4),
                                     "calibration_plus_band": round(cal_plan["estimated_usd"] + bl_plan["estimated_usd"], 4)},
                "pricing": "input ceil(prompt chars / 2.2) + 331 tokens, output 40 tokens per pair, Batch rates "
                           "0.05 / 0.25 USD per million (omission.judge.HAIKU_RATES, headline.requests.request_price)",
                "caveat": "the real absent rate is likely about 85 in 100 (see base_rate in calibration_v2.json), "
                          "so an accurate Haiku judge would likely fail the absent-rate bar as well",
                "needs": "Varma's yes before any batch is submitted"}
        (out / "haiku_plan.json").write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    else:
        cands.sort(key=lambda nr: (-nr[1]["calibration"]["balanced_accuracy"], nr[1].get("full_run_hours") or 0.0))
        best, br = cands[0]
        mode = "embedding" if best == "embedding" else ("band" if best.endswith("_band") else "full")
        swapped = None
        if mode == "full" and (br.get("full_run_hours") or 0) > MAX_FULL_HOURS:
            alt = f"{best}_band"
            swapped = {"from": best, "full_run_hours": br["full_run_hours"], "to": alt,
                       "alt_passes": bool(res.get(alt, {}).get("passes"))}
            if res.get(alt, {}).get("passes"):
                best, br, mode = alt, res[alt], "band"
            else:
                passing = [n for n, r in cands if (r.get("full_run_hours") or 0) <= MAX_FULL_HOURS]
                if passing:
                    best, br = passing[0], res[passing[0]]
                    mode = "embedding" if best == "embedding" else ("band" if best.endswith("_band") else "full")
        choice = {"status": "passed", "judge": best, "mode": mode,
                  "model": None if mode == "embedding" else MODELS[best.replace("_band", "")],
                  "tau": tau, "tau_low": lo, "tau_high": hi, "swapped_for_runtime": swapped,
                  "ranking": [n for n, _ in cands]}
    summary = {"schema": "lens.flip.exploratory/1", "exploratory": True, "version": "5-exploratory judge2",
               "prompt_version": PROMPT_VERSION, "prompt_sha256": _sha(build_prompt("{NOTE}", "{UNIT}")),
               "format": JUDGE_FORMAT, "options": {"temperature": 0, "seed": 0, "num_ctx": NUM_CTX,
                                                   "num_predict": NUM_PREDICT, "think": False},
               "pass_bar": {"balanced_accuracy_min": PASS_BALANCED_ACCURACY, "absent_rate_real_below": PASS_ABSENT_RATE},
               "hand_labels": {k: v for k, v in hl.items() if k != "labels"},
               "recipe": recipe, "band": band_info, "tau_kappa": tk, "judges": res, "choice": choice,
               "base_rate": base_rate,
               "api_calls": 0, "seconds_total": round(time.monotonic() - t_all, 1)}
    (out / "calibration_v2.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n",
                                             encoding="utf-8")
    _jsonl(out / "calibration_decisions.jsonl", decisions)
    _jsonl(out / "real_sample_decisions.jsonl", sample_rows)
    _print_table(summary)
    _log(f"wrote {out / 'calibration_v2.json'}")
    return 0 if choice["status"] == "passed" else 2


def _c4_block(root: Path, hs, rp: Sequence[dict], conveyed: Sequence[bool | None]) -> dict[str, Any]:
    from .analysis import NStats, base_rows, study_c4
    cases, rows = base_rows(root)
    states: dict[tuple[str, str], dict] = {}
    for p, c in zip(rp, conveyed):
        states.setdefault((p["case_id"], p["system"]), {"out": {"conveyed": []}})["out"]["conveyed"].append(c)
    for r in rows:
        n = len(r["_h"].view.units)
        got = states[(r["case_id"], r["system"])]["out"]["conveyed"]
        if len(got) != n:
            raise RuntimeError(f"{r['case_id']} {r['system']}: {len(got)} answers for {n} units")
    st = NStats(cases, rows)
    return study_c4(st, rows, states)


def cmd_run(args: argparse.Namespace) -> int:
    root = Path(args.root)
    out = _out(root)
    calp = out / "calibration_v2.json"
    if not calp.exists():
        _log("refused: run calibrate first")
        return 1
    calj = json.loads(calp.read_text())
    ch = calj["choice"]
    accepted = None
    if args.accept:
        if args.accept not in calj["judges"] or args.accept == "haiku_v1":
            _log(f"refused: --accept must name a local judge of calibration_v2.json, not {args.accept!r}")
            return 1
        name = args.accept
        accepted = {"judge": name, "passes": calj["judges"][name]["passes"], "reason": args.accept_reason,
                    "note": "run on an explicit --accept; the calibrated choice was " + ch["status"]}
    elif ch["status"] != "passed":
        _log("refused: no local judge passed calibration. See the Haiku plan in calibration_v2.json. "
             "A local judge runs only with --accept NAME --accept-reason TEXT (Varma's decision).")
        return 1
    else:
        name = ch["judge"]
    if accepted is not None and not args.accept_reason:
        _log("refused: --accept needs --accept-reason (who decided, and why)")
        return 1
    mode0 = "embedding" if name == "embedding" else ("band" if name.endswith("_band") else "full")
    mode = args.mode or mode0
    model = MODELS[args.model] if args.model else (None if name == "embedding" else MODELS[name.replace("_band", "")])
    t0 = time.monotonic()
    cache = CallCache(out / "cache.sqlite")
    hs, rp, cos = _real(root, _embedder(cache, args.ollama_url))
    tau, lo, hi = calj["tau_kappa"]["tau"], calj["band"]["tau_low"], calj["band"]["tau_high"]
    emb_pred = [c >= tau for c in cos]
    final: list[bool | None]
    src: list[str]
    judged: dict[int, dict] = {}
    if mode == "embedding":
        final, src = list(emb_pred), ["embedding"] * len(rp)
    else:
        idx = list(range(len(rp))) if mode == "full" else [i for i, c in enumerate(cos) if cal.in_band(c, lo, hi)]
        if args.limit:
            idx = idx[: args.limit]
        _log(f"{model}: judging {len(idx)} of {len(rp)} real pairs ({mode})")
        j = OllamaJudge2(model, cache=cache, url=args.ollama_url, workers=args.workers)
        got = j.many([(rp[i]["note"], rp[i]["unit"]) for i in idx], _log)
        judged = dict(zip(idx, got))
        if args.limit:
            _log(f"limit {args.limit}: {j.real_calls} new calls in {j.seconds:.1f} s; no summary written")
            return 0
        final, src = [], []
        for i, c in enumerate(cos):
            if i in judged:
                final.append(judged[i]["conveyed"])
                src.append("judge")
            elif hi is not None and c >= hi:
                final.append(True)
                src.append("embedding_high")
            else:
                final.append(False)
                src.append("embedding_low")
    _jsonl(out / "decisions.jsonl", [
        {"case_id": p["case_id"], "system": p["system"], "unit_index": p["unit_index"], "cosine": round(c, 6),
         "source": s, "conveyed": f, "status": judged[i]["status"] if i in judged else None,
         "quote": judged[i]["quote"] if i in judged else None}
        for i, (p, c, s, f) in enumerate(zip(rp, cos, src, final))])
    blk = _c4_block(root, hs, rp, final)
    emb_blk = _c4_block(root, hs, rp, emb_pred)
    v5 = json.loads((root / "summary_v5.json").read_text()) if (root / "summary_v5.json").exists() else {}
    dt = time.monotonic() - t0
    s = {"schema": "lens.flip.exploratory/1", "exploratory": True, "version": "5-exploratory judge2",
         "note": "Exploratory. C4 rerun with the local judge v2. summary_v5.json and the C4 arm dir are unchanged.",
         "judge": {"choice": name, "mode": mode, "model": None if mode == "embedding" else model,
                   "prompt_version": PROMPT_VERSION, "prompt_sha256": calj["prompt_sha256"], "tau": tau,
                   "tau_low": lo, "tau_high": hi, "calibration": calj["judges"][name],
                   "calibrated_choice_status": ch["status"], "accepted_override": accepted},
         "calibration_set_sha256": calj["recipe"]["sha256"],
         "pairs": {"n": len(rp), "judged_by_model": len(judged),
                   "by_source": {k: src.count(k) for k in sorted(set(src))},
                   "unparsed": sum(f is None for f in final),
                   "unsupported_yes": sum(v["status"] == "unsupported" for v in judged.values())},
         "C4_semantic_absence_judge2": blk,
         "C4_semantic_absence_embedding_only": emb_blk,
         "C4_semantic_absence_v5_haiku": v5.get("C4_semantic_absence"),
         "api_calls": 0, "runtime_seconds": round(dt, 1)}
    (out / "summary_v5_judge2.json").write_text(json.dumps(s, indent=1, sort_keys=True, default=str) + "\n",
                                                encoding="utf-8")
    a = blk["h2c_auroc_semantic_omission"]
    _log(f"mean semantic omission {blk['mean_semantic_omission']['value']:.3f}; AUROC {a['auroc']:.3f} {a['ci95']}; "
         f"runtime {dt / 60:.1f} min")
    _log(f"wrote {out / 'summary_v5_judge2.json'}")
    return 0


def _f(x: Any, k: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{k}f}"


def table_rows(calj: dict, summ: dict | None) -> list[list[str]]:
    rows = []
    au = {}
    if summ:
        au[summ["judge"]["choice"]] = summ["C4_semantic_absence_judge2"]["h2c_auroc_semantic_omission"]["auroc"]
        au["embedding"] = summ["C4_semantic_absence_embedding_only"]["h2c_auroc_semantic_omission"]["auroc"]
        hv = summ.get("C4_semantic_absence_v5_haiku")
        if hv:
            au["haiku_v1"] = hv["h2c_auroc_semantic_omission"]["auroc"]
    for name, r in calj["judges"].items():
        c = r["calibration"]
        rows.append([name, r["judge"], _f(c["balanced_accuracy"]), _f(c["kappa"]),
                     _f(r["absent_rate_real"]["value"]), _f(au.get(name)),
                     {True: "yes", False: "no", None: "n/a"}[r.get("passes")]])
    return rows


def _print_table(calj: dict, summ: dict | None = None) -> None:
    print("| name | judge | balanced accuracy | kappa | absent rate (real) | AUROC on flips | passes |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for r in table_rows(calj, summ):
        print("| " + " | ".join(r) + " |")
    print(f"choice: {json.dumps(calj['choice'], default=str)[:400]}")


def cmd_report(args: argparse.Namespace) -> int:
    out = Path(args.root) / OUT2_NAME
    calj = json.loads((out / "calibration_v2.json").read_text())
    sp = out / "summary_v5_judge2.json"
    summ = json.loads(sp.read_text()) if sp.exists() else None
    _print_table(calj, summ)
    if summ:
        b = summ["C4_semantic_absence_judge2"]
        print(f"mean semantic omission {_f(b['mean_semantic_omission']['value'])} {b['mean_semantic_omission']['ci95']}")
        print(f"AUROC sem_om {_f(b['h2c_auroc_semantic_omission']['auroc'])} {b['h2c_auroc_semantic_omission']['ci95']}"
              f" vs half {b['h2c_auroc_semantic_omission']['diff_vs_half_ci95']}")
        print(f"p_v3 minus sem_om {_f(b['p_minus_semantic_omission']['diff'])} {b['p_minus_semantic_omission']['ci95']}")
        print(f"D* recall flips {_f(b['dstar_semantic_recall_on_flips']['value'])} "
              f"non-flips {_f(b['dstar_semantic_recall_on_nonflips']['value'])}")
        print(f"runtime {summ['runtime_seconds'] / 60:.1f} min; pairs {summ['pairs']}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lens.flip.explore5.judge2", description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(OUT))
    ap.add_argument("--ollama-url", default=OLLAMA_URL)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("calibrate")
    p.add_argument("--models", default="qwen30,qwen8")
    p.add_argument("--workers", type=int, default=WORKERS)
    p.add_argument("--rebuild", action="store_true", help="overwrite a calibration set that differs")
    p.set_defaults(func=cmd_calibrate)
    p = sub.add_parser("run")
    p.add_argument("--mode", choices=("embedding", "band", "full"), default=None, help="override the calibrated choice")
    p.add_argument("--model", choices=tuple(MODELS), default=None)
    p.add_argument("--workers", type=int, default=WORKERS)
    p.add_argument("--limit", type=int, default=0, help="judge only the first N pairs (rate check); no summary")
    p.add_argument("--accept", default=None, help="run this local judge although the calibrated choice failed")
    p.add_argument("--accept-reason", default=None, help="who decided and why (recorded in the summary)")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("report")
    p.set_defaults(func=cmd_report)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
