# ---- 6. Cost per option: $/video-hour from NB2's processing_cost. MEASURED = CPU throughput from data/smoke/timings.json; ASSUMED = GPU speed-up and both prices ----
# Options: the T2-certified config; strided variants from the grid (uncertified: not in Lambda-hat); the recall-first best-effort config (T1, uncertified); NB2's naive baseline.
p2 = LTT_MAIN["T2"]["p_test"]; lam2 = LTT_MAIN["T2"]["lam_hat"]
def opt_row(label, j, status):
    c = processing_cost(pc_of(ALL_KEYS[j]), THROUGHPUT, PRICE)
    return dict(option=label, config=describe_key(ALL_KEYS[j]), status=status, cal2_miss_synth=round(float(p2["miss"][j]), 4), cal2_pooled_precision_synth=round(float(p2["pooled_prec"][j]), 3),
                cpu_usd_per_video_hour=round(c["cpu_usd_per_video_hour"], 3), gpu_usd_per_video_hour_assumed=round(c["gpu_usd_per_video_hour"], 3))
cost_rows = [opt_row("T2 certified (chosen)", J_T2, "CERTIFIED (LTT T2, delta=0.10)")]
for j in np.where(lam2)[0]:
    if j != J_T2:
        cost_rows.append(opt_row("T2 certified (other)", int(j), "CERTIFIED (LTT T2, delta=0.10)"))
for j in sorted([j for j in range(G) if ALL_KEYS[j][1] > 1], key=lambda j: p2["p_max"][j])[:4]:
    cost_rows.append(opt_row(f"stride {ALL_KEYS[j][1]} variant", int(j), f"not certified (CAL2 p_max {p2['p_max'][j]:.3f}; not reached by the walk)" if p2["p_max"][j] <= DELTA else f"not certified (CAL2 p_max {p2['p_max'][j]:.2f})"))
cost_rows.append(opt_row("best-effort recall-first (T1)", J_BEST, "not certified (T1 empty; audit shows Prop. 26 recall >= %.3f)" % (1 - AUD["BEST"]["U"])))
COST_DF = pd.DataFrame(cost_rows)
pd.set_option("display.max_colwidth", 70); pd.set_option("display.width", 300)
print("Cost per option (synthetic recall/precision on CAL2; CPU $/h = MEASURED throughput x ASSUMED price; GPU $/h = ASSUMED speed-up x ASSUMED price):")
print(COST_DF.to_string(index=False))

# break-even GPU: the GPU column is cheaper only if speed-up > GPU price / CPU price (ASSUMED both)
BE_SPEEDUP = ASSUMED_GPU_USD_PER_HOUR / ASSUMED_CPU_USD_PER_HOUR
BE_GPU_PRICE = ASSUMED_GPU_SPEEDUP * ASSUMED_CPU_USD_PER_HOUR
print(f"\nbreak-even (ASSUMED numbers): GPU needs speed-up > GPU price / CPU price = {BE_SPEEDUP:.0f}x at ${ASSUMED_GPU_USD_PER_HOUR}/h vs ${ASSUMED_CPU_USD_PER_HOUR}/h (assumed speed-up {ASSUMED_GPU_SPEEDUP:.0f}x => GPU is "
      f"{'cheaper' if ASSUMED_GPU_SPEEDUP > BE_SPEEDUP else 'MORE expensive'} under THESE placeholders); or, at {ASSUMED_GPU_SPEEDUP:.0f}x, the GPU price must be < ${BE_GPU_PRICE:.2f}/h.")
c_be = processing_cost(dict(stride=1), dict(THROUGHPUT, assumed_gpu_speedup=BE_SPEEDUP), PRICE)
check("6.1", "processing_cost is strictly decreasing in stride (1 > 2 > 4) for both CPU and GPU columns (the stride lever exists; a-priori property)",
      COST_BY_STRIDE[1]["cpu_usd_per_video_hour"] > COST_BY_STRIDE[2]["cpu_usd_per_video_hour"] > COST_BY_STRIDE[4]["cpu_usd_per_video_hour"], f"CPU $/h {[round(COST_BY_STRIDE[s]['cpu_usd_per_video_hour'], 3) for s in (1, 2, 4)]}")
check("6.2", "break-even identity: at GPU speed-up = GPU price / CPU price the GPU and CPU $/video-hour are equal (a-priori identity of processing_cost)",
      abs(c_be["gpu_usd_per_video_hour"] - c_be["cpu_usd_per_video_hour"]) < 1e-9, f"{c_be['gpu_usd_per_video_hour']:.6f} vs {c_be['cpu_usd_per_video_hour']:.6f}")

# sensitivity of the cheapest CERTIFIED option (stride 1 here) to the ASSUMED parameters, and to the measured smoke-run variance
sens_rows = []
for sp_ in (2.0, 4.0, 8.0, 16.0, 32.0):
    for gp in (0.30, 0.60, 1.20):
        c = processing_cost(dict(stride=1), dict(THROUGHPUT, assumed_gpu_speedup=sp_), dict(assumed_cpu_usd_per_hour=ASSUMED_CPU_USD_PER_HOUR, assumed_gpu_usd_per_hour=gp))
        sens_rows.append(dict(gpu_speedup_assumed=sp_, gpu_usd_per_hour_assumed=gp, cpu_usd_per_video_hour=round(c["cpu_usd_per_video_hour"], 3),
                              gpu_usd_per_video_hour=round(c["gpu_usd_per_video_hour"], 3), cheaper="GPU" if c["gpu_usd_per_video_hour"] < c["cpu_usd_per_video_hour"] else "CPU"))
SENS_COST_DF = pd.DataFrame(sens_rows)
print("\nsensitivity of the certified option's cost (stride 1) to the ASSUMED GPU speed-up and price (CPU price held at the assumed $0.05/h):")
print(SENS_COST_DF.pivot(index="gpu_speedup_assumed", columns="gpu_usd_per_hour_assumed", values="gpu_usd_per_video_hour").to_string())
print(f"(CPU column for comparison: ${COST_BY_STRIDE[1]['cpu_usd_per_video_hour']:.3f}/video-hour, independent of the GPU assumptions)")
# smoke-run variance: NB2 recorded decode 7-42 fps and libx264 encode 0.6-4.5 fps across runs on this shared VM (several-fold); scenario costs at stride 1 and stride 4
VAR = {"slow run (decode 7 fps, encode 0.6 fps)": dict(decode_fps=7.0, encode_fps=0.6), "as measured in timings.json": {}, "fast run (decode 42 fps, encode 4.5 fps)": dict(decode_fps=42.0, encode_fps=4.5)}
var_rows = []
for nm, kw in VAR.items():
    for s in (1, 4):
        c = processing_cost(dict(stride=s), dict(THROUGHPUT, **kw), PRICE)
        var_rows.append(dict(scenario=nm, stride=s, cpu_usd_per_video_hour=round(c["cpu_usd_per_video_hour"], 3)))
VAR_DF = pd.DataFrame(var_rows)
print("\nsmoke-timing variance (ranges recorded in Notebook 2 across runs on this shared WSL2 VM; detect fps held at the measured value):"); print(VAR_DF.to_string(index=False))
_ratio = VAR_DF[VAR_DF["stride"] == 1]["cpu_usd_per_video_hour"].max() / VAR_DF[VAR_DF["stride"] == 1]["cpu_usd_per_video_hour"].min()
check("6.3", "the run-to-run variance of the measured smoke timings moves the CPU $/video-hour by >= 2x (a-priori: so no single $ figure should be quoted to the client)",
      _ratio >= 2.0, f"max/min = {_ratio:.3f}x at stride 1 (threshold >= 2.0; borderline if close)")
