# ---- (c) Thm 19: fixed-sequence certification along a PRE-REGISTERED nested family [P3 Sec. 4.8, Thm 19, facered-q9] ----
# Family: score threshold 0.40 ... 0.90 (11 prefixes), everything else fixed. g_1 = thr 0.90 (least inclusive, largest excluded pool), g_M = thr 0.40 (most inclusive).
# Order of testing (paper): g_M, g_{M-1}, ..., g_1; stop at the first failure; return the LAST prefix that passed. Each test runs at the full level delta (no delta/M).
t0 = time.time()
NEST = {}
for key in NESTED_KEYS:
    bb = run_pipeline(DETS["AUDIT"], CORP["AUDIT"], pc_of(key)); res = evaluate(CORP["AUDIT"], bb)
    pi = res["per_instance"].sort_values("track_id").reset_index(drop=True)
    it = segment_items(CORP["AUDIT"], bb)
    NEST[key] = dict(missed=pi["instance_missed"].to_numpy(), excl=(~it["has_box"]).to_numpy(), pos=it["positive"].to_numpy(),
                     precision=res["precision"], n_box=res["n_blur_boxframes"])
print(f"nested family evaluated on the sealed AUDIT in {time.time() - t0:.1f}s")

# nesting check: as the threshold falls, blur sets should only grow
inst_viol = sum(int((NEST[NESTED_KEYS[i]]["missed"] & ~NEST[NESTED_KEYS[i + 1]]["missed"]).sum()) for i in range(len(NESTED_KEYS) - 1))   # missed at lower thr but covered at higher thr
pool_viol = sum(int((NEST[NESTED_KEYS[i]]["excl"] & ~NEST[NESTED_KEYS[i + 1]]["excl"]).sum()) for i in range(len(NESTED_KEYS) - 1))       # excluded at lower thr but boxed at higher thr
print(f"nesting on the AUDIT: violations (instance covered at a higher threshold but missed at a lower one): {inst_viol} of {len(NEST[NESTED_KEYS[0]]['missed'])} instances x {len(NESTED_KEYS) - 1} steps; "
      f"segment-pool violations (excluded at lower thr but boxed at higher thr): {pool_viol} of {len(NEST[NESTED_KEYS[0]]['excl'])} segments x {len(NESTED_KEYS) - 1} steps")
check("5.5", "nesting assumption of Thm 19: over the 10 threshold steps, NO audited instance is missed at a lower threshold yet covered at a higher one, and no segment is excluded at a "
      "lower threshold yet boxed at a higher one (a-priori: 0 violations = exactly nested; otherwise the nesting is only approximate)",
      inst_viol == 0 and pool_viol == 0, f"{inst_viol} instance-level and {pool_viol} pool-level violations")

# (c1) Thm 19 as stated: missed MASS of the excluded segment pools, epsilon = 0.01 (pre-registered), n_m = 300 fresh draws per prefix at level delta = 0.05
rng19 = np.random.default_rng(SEED + 22)
N_SEGS = len(ITEMS)
rows19, walk_flags = [], []
for key in NESTED_KEYS:                                       # order g_M ... g_1 = threshold 0.40 -> 0.90
    d = NEST[key]; exc_idx = np.where(d["excl"])[0]; N0m = len(exc_idx)
    draw_m = rng19.choice(N0m, size=N0_SAMPLE, replace=False)
    Km = int(d["pos"][exc_idx[draw_m]].sum())
    MU = hypergeom_upper(Km, N0m, N0_SAMPLE, DELTA_AUDIT)
    passed_seg = (MU / N_SEGS) <= EPS_SEG
    Cm = int(d["missed"].sum()); Qm = len(d["missed"])
    Um = float(cp_upper(Cm, Qm, DELTA_AUDIT)); passed_inst = Um <= EPS_INST
    rows19.append(dict(thr=key[0], excluded_pool_N0=N0m, K_m=Km, r_bound_seg=round(MU / N_SEGS, 4), passes_seg_eps=bool(passed_seg),
                       instance_miss_rate=round(Cm / Qm, 4), U_Q_C_inst=round(Um, 4), passes_inst_eps=bool(passed_inst), pooled_precision=round(d["precision"], 3)))
T19 = pd.DataFrame(rows19)
def walk(flags):
    ret = -1
    for m, f in enumerate(flags):                            # test order g_M -> g_1 = index 0 -> M-1 in NESTED_KEYS
        if not f:
            break
        ret = m
    return ret
ret_seg, ret_inst = walk(T19["passes_seg_eps"].tolist()), walk(T19["passes_inst_eps"].tolist())
# union-bound (delta/M) variant of the INSTANCE-level test, for the "no delta/M penalty" comparison
T19["passes_inst_eps_deltaM"] = [bool(float(cp_upper(int(NEST[k]["missed"].sum()), len(NEST[k]["missed"]), DELTA_AUDIT / len(NESTED_KEYS))) <= EPS_INST) for k in NESTED_KEYS]
pd.set_option("display.width", 260)
print(); print(T19.to_string(index=False))
print(f"\nThm 19 (missed MASS, eps_seg = {EPS_SEG}): returned prefix = thr {NESTED_THR[ret_seg] if ret_seg >= 0 else 'none'} "
      f"(the LEAST inclusive prefix passes: every excluded pool holds ~no positive segments)")
print(f"   ... yet that prefix misses {T19.loc[ret_seg, 'instance_miss_rate']:.1%} of audited face INSTANCES (pooled precision {T19.loc[ret_seg, 'pooled_precision']}): "
      f"the segment-level certificate is about a different, much weaker event (see Remark 9 above).")
print(f"Thm 19 instance-level twin (DERIVED HERE: same walk and level delta, pass_m = [U_Q(C_m, delta) <= eps_inst = {EPS_INST}]): returned prefix = thr "
      f"{NESTED_THR[ret_inst] if ret_inst >= 0 else 'none'};  with a delta/M union bound the walk would stop at thr "
      f"{NESTED_THR[walk(T19['passes_inst_eps_deltaM'].tolist())] if walk(T19['passes_inst_eps_deltaM'].tolist()) >= 0 else 'none'}")
_mono = bool((np.diff(T19["instance_miss_rate"].to_numpy()) >= -1e-12).all())
check("5.6", "the audited instance miss rate is non-decreasing along the nested family (0.40 -> 0.90), as nested blur sets imply (a-priori)", _mono,
      f"miss rates {T19['instance_miss_rate'].tolist()}")
