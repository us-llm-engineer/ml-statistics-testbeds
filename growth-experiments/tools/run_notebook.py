"""Execute a .ipynb in place with a local ipykernel (no nbconvert/nbclient needed).

Usage: /usr/local/bin/python3 tools/run_notebook.py notebooks/NAME.ipynb [--timeout 1800]
Runs every code cell top to bottom with cwd = the notebook's folder, stores outputs
(stream, execute_result, display_data incl. image/png, error) back into the file, and
prints one line per cell plus a final summary: errors, and code cells whose source
starts with '# Visualization only' but produced no image/png.
"""
import json, sys, os, time, argparse, queue
from jupyter_client import KernelManager

ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--timeout", type=int, default=1800)
a = ap.parse_args()
path = os.path.abspath(a.path)
nb = json.load(open(path))
km = KernelManager(kernel_cmd=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"])
km.start_kernel(cwd=os.path.dirname(path))
kc = km.client(); kc.start_channels(); kc.wait_for_ready(timeout=60)
errors, missing_img, count = [], [], 0
t0 = time.time()
try:
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        count += 1
        msg_id = kc.execute(src)
        outputs, has_img, err = [], False, None
        while True:
            try:
                msg = kc.get_iopub_msg(timeout=a.timeout)
            except queue.Empty:
                err = "TIMEOUT"; break
            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            t, c = msg["msg_type"], msg["content"]
            if t == "status" and c["execution_state"] == "idle":
                break
            if t == "stream":
                if outputs and outputs[-1].get("output_type") == "stream" and outputs[-1]["name"] == c["name"]:
                    outputs[-1]["text"] += c["text"]
                else:
                    outputs.append({"output_type": "stream", "name": c["name"], "text": c["text"]})
            elif t in ("execute_result", "display_data"):
                o = {"output_type": t, "data": c["data"], "metadata": c.get("metadata", {})}
                if t == "execute_result":
                    o["execution_count"] = c.get("execution_count")
                has_img = has_img or "image/png" in c["data"]
                outputs.append(o)
            elif t == "error":
                outputs.append({"output_type": "error", "ename": c["ename"], "evalue": c["evalue"], "traceback": c["traceback"]})
                err = f'{c["ename"]}: {c["evalue"]}'
        for o in outputs:
            if o["output_type"] == "stream":
                o["text"] = o["text"].splitlines(keepends=True)
        cell["outputs"] = outputs; cell["execution_count"] = count
        vis = src.lstrip().lower().startswith("# visualization only")
        if vis and not has_img:
            missing_img.append(idx)
        print(f"cell {idx:3d} {'ERR ' + err if err else 'ok'}{' [img]' if has_img else ''} ({time.time()-t0:.0f}s)", flush=True)
        if err:
            errors.append((idx, err)); break
finally:
    json.dump(nb, open(path, "w"), indent=1, ensure_ascii=False)
    kc.stop_channels(); km.shutdown_kernel(now=True)
print(f"SUMMARY errors={errors} vis_cells_without_image={missing_img} code_cells_run={count} secs={time.time()-t0:.0f}")
