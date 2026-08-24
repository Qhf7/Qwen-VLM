from __future__ import annotations

STATE_NAMES = (
    "background",
    "normal component surface",
    "normal structural edge and repeated texture",
    "normal reflection and highlight",
    "fine scratch",
    "surface crack",
    "dent",
    "stain and contamination",
    "chipped or broken region",
    "foreign object or texture anomaly",
    "discoloration",
    "deformation or misalignment",
)

BACKGROUND = 0
NORMAL_SURFACE = 1
NORMAL_EDGE = 2
NORMAL_REFLECTION = 3
FIRST_ANOMALY = 4
ANOMALY_STATES = tuple(range(FIRST_ANOMALY, len(STATE_NAMES)))

GENERATOR_TO_STATE = {
    "hairline": 4,
    "scratch": 4,
    "crack": 5,
    "dent": 6,
    "speck": 7,
    "contamination": 7,
    "chip": 8,
    "hole": 8,
    "cutpaste": 9,
    "discolour": 10,
    "misalign": 11,
}

DEFECT_GENERATORS = tuple(GENERATOR_TO_STATE)
