"""Assemble Notebook 3 from part modules.

    .venv/bin/python notebooks/build/build_nb3.py                       # full notebook (prelude + a + b + closing)
    .venv/bin/python notebooks/build/build_nb3.py --parts prelude,a --out notebooks/_test_a.ipynb   # test build

Test notebooks MUST be written to notebooks/ (same cwd as the real notebook, because Notebook 2's core cells
use paths relative to the notebooks/ folder) and must be named _test_*.ipynb; delete them when done.
"""
import argparse
import importlib
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ap = argparse.ArgumentParser()
ap.add_argument("--parts", default="prelude,a,b,c,d,b8,closing")
ap.add_argument("--out", default=str(HERE.parent / "03_project_walkthrough_part2.ipynb"))
args = ap.parse_args()

MODS = {"prelude": "nb3_prelude", "a": "nb3_part_a", "b": "nb3_part_b", "c": "nb3_part_c", "d": "nb3_part_d",
        "b8": "nb3_part_b8", "closing": "nb3_closing"}
cells = []
for p in args.parts.split(","):
    cells.extend(importlib.import_module(MODS[p]).cells())

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"name": "idp-fintech", "display_name": "idp-fintech (.venv)", "language": "python"}
nbf.write(nb, args.out)
print(f"wrote {args.out} with {len(cells)} cells")
