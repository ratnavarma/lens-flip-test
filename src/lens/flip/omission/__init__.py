"""Exploratory omission comparator for H2c (design section 7.3).

Outside frozen protocol v3. Nothing here changes summary_v3.json. The frozen omission baseline (units of V not
quoted verbatim in N) is 1.0 on almost every note. This package builds a non-verbatim comparator instead:
a unit of V is conveyed by the note when its best cosine to a note sentence reaches a threshold calibrated on
Varma's hand labels (and, in the embedding+judge mode, when a local judge says so on borderline pairs).

Modules: frame (the 490 notes and their units), coverage (embedding coverage), sample (calibration set),
labels (parser for the filled file), calibrate (thresholds and agreement metrics), judge (local judge and the
Haiku plan), runner (the run), cli.
"""

SCHEMA = "lens.flip.omission/1"
VERSION = "omission-explore-v1"
SEED = 21
OUT_DIR = "results/omission-explore-v1"
SUMMARY_NAME = "summary_omission_v1.json"
SUMMARY_JUDGE_NAME = "summary_omission_v1_judge.json"
KEY_NAME = "calibration_key.jsonl"
LABELS_NAME = "labels.jsonl"
VAULT_FILE = "results/omission-explore-v1/omission-calibration.md"
