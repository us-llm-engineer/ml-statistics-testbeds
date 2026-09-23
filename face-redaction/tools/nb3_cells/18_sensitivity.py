# ---- 3(e) "What detector quality would make T1 certifiable?" (derived here, EXPLORATORY): shift the simulator's detector calibration ----
# Levels change `det` keys of NB2's simulate_detector (base = face-logit intercept, fp_gain = false-positive logit scale, clutter rate) and, for L2/L3, the
# screen/photo policy (apply_policy: TV/photo faces become faces to redact). A small core grid (thr x pad at gap=30, plus strided configs) is re-evaluated.
SENS_KEYS = [(t, 1, 30, p, 0.3) for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95) for p in (5, 20, 40)] + \
            [(t, s, 30, 40, 0.3) for t in (0.7, 0.9) for s in (2, 4)]
SENS_LEVELS = [
    ("L0 baseline detector (default)", {}, False),
    ("L1 better detector: base 2.0, fp_gain 1.5, clutter 0.2", dict(base=2.0, fp_gain=1.5, clutter_rate_per_1000frames=0.2), False),
    ("L2 = L1 + redact screen/photo faces (policy)", dict(base=2.0, fp_gain=1.5, clutter_rate_per_1000frames=0.2), True),
    ("L3 much better detector: base 3.0, fp_gain 2.5, clutter 0.05, + policy", dict(base=3.0, fp_gain=2.5, clutter_rate_per_1000frames=0.05), True),
    ("L4 excellent detector: base 4.0, fp_gain 3.0, clutter 0.02, + policy", dict(base=4.0, fp_gain=3.0, clutter_rate_per_1000frames=0.02), True),
]
SENS_DELTA = DELTA / (len(SENS_LEVELS) - 1)      # exploratory levels L1-L4 share the delta budget (Bonferroni), all tested on the already-opened CAL
SENS_ROWS, SENS_DETAIL = [], {}
t0 = time.time()
for li, (label, det_kw, redact) in enumerate(SENS_LEVELS):
    if li == 0:
        continue                                     # L0 = the main run above (T1 empty on CAL1); not re-run
    for sp in ("TUNE", "CAL"):
        _c = apply_policy(CORP[sp], True) if redact else CORP[sp]
        _d = simulate_detector(_c, det_kw, seed=SEED + 30 + li)
        register_work(f"S{li}{sp}", _c, _d)
    res = eval_many([(f"S{li}{sp}", k) for sp in ("TUNE", "CAL") for k in SENS_KEYS])
    Mt = {sp: to_matrices({(f"S{li}{sp}", k): res[(f"S{li}{sp}", k)] for k in SENS_KEYS}, [f"S{li}{sp}"], SENS_KEYS) for sp in ("TUNE", "CAL")}
    it, ic = np.arange(Mt["TUNE"]["n_inst"].size), np.arange(Mt["CAL"]["n_inst"].size)
    lam, det = ltt_select(Mt["TUNE"], it, Mt["CAL"], ic, ALPHA_T1, SENS_DELTA, starts=(0,))
    pc_ = det["p_test"]
    j_best = int(np.argmin(np.maximum(pc_["miss"] / ALPHA_T1["miss"], pc_["fb"] / ALPHA_T1["fb"])))
    both = (pc_["miss"] <= ALPHA_T1["miss"]) & (pc_["fb"] <= ALPHA_T1["fb"])
    cert = np.where(lam)[0]
    cheapest = min(cert, key=lambda j: (round(cost_of_key(SENS_KEYS[j]), 9), pc_["fb"][j])) if len(cert) else None
    rate = float(pc_["miss"][j_best])
    n_needed = next((n for n in range(50, 20001, 25) if rate < ALPHA_T1["miss"] and p_binom(int(round(rate * n)), n, ALPHA_T1["miss"]) <= SENS_DELTA), None)
    SENS_ROWS.append(dict(level=label, T1_certified=bool(len(cert)), n_certified=int(len(cert)),
                          n_configs_meeting_T1_by_point_estimate=int(both.sum()), best_config=describe_key(SENS_KEYS[j_best]),
                          best_cal_miss=round(float(pc_["miss"][j_best]), 4), best_cal_fdp=round(float(pc_["fb"][j_best]), 4),
                          best_cal_pooled_precision=round(float(pc_["pooled_prec"][j_best]), 3),
                          instances_needed_to_certify_that_miss_rate=(n_needed if n_needed else "not reachable"),
                          cheapest_certified=(describe_key(SENS_KEYS[cheapest]) if cheapest is not None else "none (abstain)"),
                          cheapest_cpu_usd_per_h_assumed_price=(round(cost_of_key(SENS_KEYS[cheapest]), 3) if cheapest is not None else float("nan"))))
    SENS_DETAIL[li] = dict(lam=lam, p_test=pc_, Mt=Mt)
    print(f"  {label}: done ({time.time() - t0:.0f}s elapsed)")
SENS_DF = pd.DataFrame(SENS_ROWS)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
print(); print(SENS_DF.to_string(index=False))
_exc = [max(r["best_cal_miss"] / ALPHA_T1["miss"], r["best_cal_fdp"] / ALPHA_T1["fb"]) for r in SENS_ROWS]
print(f"\nT1 certified at L1..L4: {[r['T1_certified'] for r in SENS_ROWS]}; best-config normalised excess over T1 (max of miss/0.01, FDP/0.05; <=1 means both targets met by point estimate): {np.round(_exc, 2).tolist()}")
check("3.4", "the sensitivity levels really are ordered by detector quality: the best CAL point-estimate normalised excess over T1 is non-increasing from L1 to L4 (a-priori claim)",
      all(_exc[i + 1] <= _exc[i] + 1e-9 for i in range(len(_exc) - 1)), f"excess {np.round(_exc, 2).tolist()}")
