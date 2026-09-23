"""Execute a notebook in place on the face-redaction kernel and report errors + image counts.

Usage: .venv/bin/python tools/run_notebook.py notebooks/01_research_foundations.ipynb [timeout_s]
"""
import sys, nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

path = sys.argv[1]
timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 1800
nb = nbformat.read(path, as_version=4)
client = NotebookClient(nb, timeout=timeout, kernel_name="face-redaction", resources={"metadata": {"path": "notebooks"}})
status = 0
try:
    client.execute()
except CellExecutionError as e:
    print("EXECUTION ERROR:\n", str(e)[:3000])
    status = 1
finally:
    nbformat.write(nb, path)
code = [c for c in nb.cells if c.cell_type == "code"]
errors = sum(1 for c in code for o in c.get("outputs", []) if o.get("output_type") == "error")
images = sum(1 for c in code for o in c.get("outputs", []) if "image/png" in o.get("data", {}))
viz = [i for i, c in enumerate(nb.cells) if c.cell_type == "code" and "# VIZ" in c.source]
viz_missing = [i for i in viz if not any("image/png" in o.get("data", {}) for o in nb.cells[i].get("outputs", []))]
print(f"cells={len(nb.cells)} code={len(code)} errors={errors} png_outputs={images} viz_cells={len(viz)} viz_without_image={viz_missing}")
sys.exit(status or (1 if errors or viz_missing else 0))
