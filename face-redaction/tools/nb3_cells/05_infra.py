# ---- per-episode loss cache + a small fork-based parallel evaluator (all built on NB2's run_pipeline + evaluate) ----
import os

def pc_of(key: tuple) -> dict:
    """key = (score_thr, stride, max_gap, pad, margin)  ->  NB2 pipeline-config dict (pad_before = pad_after = pad)."""
    thr, stride, gap, pad, margin = key
    return dict(score_thr=thr, stride=int(stride), max_gap=int(gap), pad_before=int(pad), pad_after=int(pad), margin=margin)

def episode_losses(corpus: dict, detections: pd.DataFrame, pc: dict) -> dict:
    """run_pipeline + evaluate on `corpus`, then reduce to per-EPISODE integer counts aligned with corpus['episodes']:
    n_inst (face-to-redact instances), n_miss (instances with >= 1 uncovered visible frame, strict rule), n_box (blur box-frames),
    n_false (blur box-frames overlapping no face-to-redact GT box). Episodes are independent, so counts add across episodes."""
    bb = run_pipeline(detections, corpus, pc).reset_index(drop=True)
    res = evaluate(corpus, bb)
    eids = corpus["episodes"]["episode_id"].to_numpy()
    pos = pd.Series(np.arange(len(eids)), index=eids)
    pi = res["per_instance"]
    n_inst = np.bincount(pos[pi["episode_id"]].to_numpy(), minlength=len(eids))
    n_miss = np.bincount(pos[pi["episode_id"]].to_numpy(), weights=pi["instance_missed"].to_numpy().astype(float), minlength=len(eids))
    if len(bb):
        bpos = pos[bb["episode_id"]].to_numpy()
        n_box = np.bincount(bpos, minlength=len(eids))
        n_false = np.bincount(bpos, weights=res["blur_is_false"].astype(float), minlength=len(eids))
    else:
        n_box, n_false = np.zeros(len(eids), int), np.zeros(len(eids))
    return dict(n_inst=n_inst.astype(int), n_miss=n_miss.astype(int), n_box=n_box.astype(int), n_false=n_false.astype(int))

_WORK = {}                                   # name -> (corpus, detections); filled BEFORE a pool forks (children inherit it)
_EMPTY_BOXES = pd.DataFrame(columns=["episode_id", "frame", "x1", "y1", "x2", "y2", "track_key", "source_track_id"])

def register_work(name: str, corpus: dict, detections: pd.DataFrame):
    evaluate(corpus, _EMPTY_BOXES)           # build the ground-truth cache once in the parent (children inherit it copy-on-write)
    _WORK[name] = (corpus, detections)

def _task(args):
    name, key = args
    corpus, dets = _WORK[name]
    return episode_losses(corpus, dets, pc_of(key))

N_WORKERS = min(5, os.cpu_count() or 1)   # measured on this shared VM: more workers were SLOWER (memory pressure, swap)
def eval_many(tasks: list, workers: int = N_WORKERS) -> dict:
    """tasks: list of (work_name, config_key). Returns {(work_name, key): loss dict}. Fork pool; results are identical to serial."""
    if not tasks:
        return {}
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(workers, mp_context=ctx) as ex:
        out = list(ex.map(_task, tasks, chunksize=1))
    return dict(zip(tasks, out))

# ---- the configuration grid Lambda: (score_thr, stride, max_gap, pad, margin) ----
# NB2's grid extended toward MORE PERMISSIVE settings (lower thresholds, more bridging, more padding, larger margin). Configs with
# max_gap = 0 and thr < 0.4 are left out: they create hundreds of thousands of one-frame segments (run_pipeline cost, ~10x slower)
# and, as NB2 showed, buy recall only at near-zero precision.
GRID_KEYS = []
for _thr in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    for _gap in [10, 30]:
        for _pad in [5, 20, 40]:
            GRID_KEYS.append((_thr, 1, _gap, _pad, 0.3))                        # stride 1, margin 0.3: 36 configs
for _thr in [0.5, 0.6, 0.7, 0.8]:
    GRID_KEYS.append((_thr, 1, 30, 20, 0.1))                                    # smaller margin: 4 configs
for _stride in [2, 4]:
    for _thr in [0.5, 0.6, 0.7]:
        for _pad in [20, 40]:
            GRID_KEYS.append((_thr, _stride, 30, _pad, 0.3))                    # stride 2 / 4 (the cost knob): 12 configs
CRC_EXTRA = [(t, 1, 30, 20, 0.3) for t in [0.45, 0.55, 0.65, 0.75, 0.85, 0.95]]  # finer threshold ladder used by the SeqCRC/CRC comparison
ALL_KEYS = GRID_KEYS + CRC_EXTRA
KEY_INDEX = {k: i for i, k in enumerate(ALL_KEYS)}
G = len(ALL_KEYS)
print(f"grid Lambda: {len(GRID_KEYS)} configs (40 at stride 1, 12 at stride 2/4) + {len(CRC_EXTRA)} extra thresholds = {G}")

# ---- planning: time one config on TUNE to size the batch work ----
register_work("TUNE", CORP["TUNE"], DETS["TUNE"])
_t = []
for _k in [(0.7, 1, 30, 20, 0.3), (0.4, 1, 30, 40, 0.3), (0.6, 2, 30, 20, 0.3)]:
    t0 = time.time(); _ = episode_losses(CORP["TUNE"], DETS["TUNE"], pc_of(_k)); _t.append(time.time() - t0)
t_cfg_tune = float(np.median(_t))                # median of 3: robust to one contention stall (timings on this VM vary several-fold)
n_eps_pool = N_TUNE + N_CAL + N_TRUTH
est_cpu_s = G * t_cfg_tune * (n_eps_pool / N_TUNE)
print(f"serial per-config time on TUNE ({N_TUNE} episodes): {np.round(_t, 2)} s; est. CPU-seconds to evaluate ALL {G} configs on "
      f"TUNE+CAL+TRUTH ({n_eps_pool} episodes): {est_cpu_s:.0f} s  (~{est_cpu_s / N_WORKERS:.0f} s wall on {N_WORKERS} workers)")
check("1.4", "estimated CPU cost of the full grid on all pools <= 1,500 CPU-seconds (a-priori budget, ~4 min wall on 8 workers)",
      est_cpu_s <= 1500, f"{est_cpu_s:.0f} CPU-s")
