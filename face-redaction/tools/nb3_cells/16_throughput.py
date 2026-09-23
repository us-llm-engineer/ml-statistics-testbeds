# ---- throughput dict built at run time from the MEASURED smoke-test JSON (Notebook 2, section 4), plus ASSUMED GPU/price constants ----
_T = json.loads(TIMINGS_PATH.read_text())
def _ffmpeg_fps(step, fallback):
    v = [r["fps"] for r in _T["ffmpeg"] if r["step"] == step and r["fps"] == r["fps"]]
    return (float(v[0]), "measured") if v else (float(fallback), "ASSUMED fallback")
decode_fps_measured, _src_dec = _ffmpeg_fps("decode", 25.0)
encode_fps_measured, _src_enc = _ffmpeg_fps("encode_x264_veryfast", 2.0)
detect_fps_measured = 1000.0 / float(np.mean([r["ms"] for r in _T["detect"] if r["mode"] == "4k_full"]))
ASSUMED_GPU_SPEEDUP = 8.0          # ASSUMPTION (same name/value as NB2): GPU-batched detector ~8x faster than this CPU path
ASSUMED_CPU_USD_PER_HOUR = 0.05    # ASSUMPTION (same as NB2): generic CPU instance rental, NOT a provider quote
ASSUMED_GPU_USD_PER_HOUR = 0.60    # ASSUMPTION (same as NB2): small-GPU instance rental, order of magnitude, NOT a provider quote
THROUGHPUT = dict(decode_fps=decode_fps_measured, detect_fps=detect_fps_measured, encode_fps=encode_fps_measured, fps=30,
                  assumed_gpu_speedup=ASSUMED_GPU_SPEEDUP)
PRICE = dict(assumed_cpu_usd_per_hour=ASSUMED_CPU_USD_PER_HOUR, assumed_gpu_usd_per_hour=ASSUMED_GPU_USD_PER_HOUR)
print(f"MEASURED on this shared WSL2 VM (data/smoke/timings.json, 15-frame 4K clip; several-fold run-to-run variance, NB2 section 4): "
      f"decode {decode_fps_measured:.1f} fps ({_src_dec}), YuNet 4K detect {detect_fps_measured:.2f} fps, libx264-veryfast encode {encode_fps_measured:.2f} fps ({_src_enc})")
print(f"ASSUMED (human must replace): GPU speed-up {ASSUMED_GPU_SPEEDUP}x, CPU ${ASSUMED_CPU_USD_PER_HOUR}/h, GPU ${ASSUMED_GPU_USD_PER_HOUR}/h")
COST_BY_STRIDE = {s: processing_cost(dict(stride=s), THROUGHPUT, PRICE) for s in (1, 2, 4)}
print("processing_cost by stride (CPU $/video-hour = measured throughput x ASSUMED price):",
      {s: round(v["cpu_usd_per_video_hour"], 3) for s, v in COST_BY_STRIDE.items()})
def cost_of_key(key, kind="cpu"):
    return processing_cost(pc_of(key), THROUGHPUT, PRICE)[f"{kind}_usd_per_video_hour"]
COST_VEC = np.array([cost_of_key(k) for k in ALL_KEYS])         # CPU $/video-hour per grid config (depends on stride only)
