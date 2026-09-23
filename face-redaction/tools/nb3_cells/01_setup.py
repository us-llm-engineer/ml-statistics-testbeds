%matplotlib inline
import json
import math
import time
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")          # fork-in-multithreaded-process notices, pandas chained-assign notices
T_NOTEBOOK_START = time.time()

SEED = 20260917                            # same master seed as Notebook 2
NOTEBOOK_DIR = Path.cwd()
ROOT_DIR = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == "notebooks" else NOTEBOOK_DIR
DATA_DIR = ROOT_DIR / "data"
NB3_DIR = DATA_DIR / "nb3"                 # scratch/report files of this notebook (small)
NB3_DIR.mkdir(parents=True, exist_ok=True)
NB2_PATH = NOTEBOOK_DIR / "02_project_walkthrough_part1.ipynb"
TIMINGS_PATH = DATA_DIR / "smoke" / "timings.json"

# Palette family of Notebook 2 (same names, same meaning) plus three semantic aliases used only here.
PALETTE = {
    "slate": "#3b4a6b", "teal": "#1f8a70", "coral": "#d1495b", "amber": "#e0a458",
    "plum": "#7b5ea7", "steel": "#4c8bb4", "sand": "#c9b896", "moss": "#6b8f52",
    "ink": "#22223b", "rose": "#e07a9e", "berry": "#a8476b", "stone": "#8d99ae",
}
C_LTT, C_CRC, C_CERT, C_REQ = PALETTE["teal"], PALETTE["amber"], PALETTE["plum"], PALETTE["coral"]
KIND_COLORS = {
    "bystander_adult": "#1f8a70", "child": "#e0a458", "mirror_reflection": "#7b5ea7",
    "small_distant": "#4c8bb4", "brief_crossing": "#6b8f52", "heavily_occluded": "#d1495b",
    "extreme_angle": "#22223b",
}
plt.rcParams.update({"figure.dpi": 100, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8.5})

# Event log: the order in which design decisions are fixed and labels are opened is itself checked
# (P3 Assumption 1: the certified configuration must be fixed BEFORE the audit labels are examined).
EVENTS = []
def log_event(name):
    EVENTS.append((len(EVENTS), name, time.time()))

CHECKS = []                                # (id, description, verdict) collected for the final summary
def check(cid, description, ok, detail=""):
    """Print one numbered self-check. `ok` is decided against an a-priori threshold stated in the
    description; PASS/FAIL is never adjusted after the number is seen."""
    verdict = "PASS" if ok else "FAIL"
    CHECKS.append((cid, description, verdict))
    print(f"CHECK {cid}: {description} -> {detail} -> {verdict}")

print(f"ROOT_DIR = {ROOT_DIR}")
print(f"seed = {SEED}; nb2 notebook exists: {NB2_PATH.exists()}; timings.json exists: {TIMINGS_PATH.exists()}")
