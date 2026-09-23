"""Assemble notebooks/03_project_walkthrough_part2.ipynb from cell files.

tools/nb3_cells/<NN>_<name>.py   one code cell each, run in sorted-filename order
tools/nb3_md/before_<NN>_<name>.md / after_<NN>_<name>.md   markdown placed before / after that code cell
    (several markdown cells in one file are separated by a line containing only  <!-- cell -->)

Usage: .venv/bin/python tools/build_nb3.py [--out PATH] [--no-md]
"""
import argparse, pathlib
import nbformat

ROOT = pathlib.Path(__file__).resolve().parent.parent
CELLS, MD = ROOT / "tools" / "nb3_cells", ROOT / "tools" / "nb3_md"


def md_cells(path):
    if not path.exists():
        return []
    parts = [p.strip("\n") for p in path.read_text().split("\n<!-- cell -->\n")]
    return [nbformat.v4.new_markdown_cell(p) for p in parts if p.strip()]


def build(out, with_md=True):
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "face-redaction (.venv)", "language": "python", "name": "face-redaction"}
    for f in sorted(CELLS.glob("*.py")):
        if with_md:
            nb.cells += md_cells(MD / f"before_{f.stem}.md")
        nb.cells.append(nbformat.v4.new_code_cell(f.read_text().rstrip("\n")))
        if with_md:
            nb.cells += md_cells(MD / f"after_{f.stem}.md")
    nbformat.write(nb, out)
    return nb


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "notebooks" / "03_project_walkthrough_part2.ipynb"))
    ap.add_argument("--no-md", action="store_true")
    a = ap.parse_args()
    nb = build(a.out, with_md=not a.no_md)
    print(f"wrote {a.out}: {len(nb.cells)} cells ({sum(c.cell_type == 'code' for c in nb.cells)} code)")
