# ---- 3(c) What went wrong with T2-v1, and the revision (pre-registered on TUNE + planned sample sizes only, tested on the FRESH split CAL2) ----
_d = LTT_MAIN["T2v1"]; _j0 = int(_d["path"][0]); _pt, _pc = _d["p_tune"], _d["p_test"]
print(f"T2-v1 diagnosis. The path's first config ({describe_key(ALL_KEYS[_j0])}) is the one that looked SAFEST on TUNE: "
      f"{int(round(_pt['miss'][_j0] * _pt['n_inst']))}/{_pt['n_inst']} misses on TUNE ({_pt['miss'][_j0]:.4f}, p_miss={_pt['p_miss'][_j0]:.4f}), "
      f"but {int(round(_pc['miss'][_j0] * _pc['n_inst']))}/{_pc['n_inst']} on CAL ({_pc['miss'][_j0]:.4f}, p_miss={_pc['p_miss'][_j0]:.3f} > delta={DELTA}). "
      f"Fixed-sequence testing stops at the first failure, so Lambda-hat was empty although {int((_pc['p_max'] <= DELTA).sum())} of {G} configs have CAL p_max <= delta.")
print(f"Two reasons, both visible BEFORE looking at CAL had we done the arithmetic: (i) alpha_miss={ALPHA_T2_V1['miss']} needs the CAL miss count to fall well below "
      f"{ALPHA_T2_V1['miss'] * _pc['n_inst']:.0f} expected misses (only ~19 at n={_pc['n_inst']}); (ii) the first path element is the winner of a 58-way contest on TUNE (winner's curse).")

# --- T2-v2: rule fixed from TUNE + PLANNED CAL2 sizes only (no CAL1 or CAL2 label used) ---
n_cal2_inst_planned = int(round(S_TUNE["n_inst"] / len(IDX_TUNE) * N_CAL2))       # TUNE's instances-per-episode x planned CAL2 episodes
def choose_t2_v2(tune_miss, tune_fb, n_inst_plan, n_ep_plan, ladder=(0.05, 0.075, 0.10, 0.15, 0.20), z=2.5, min_configs=8, min_expected_misses=30):
    """alpha_miss: smallest ladder value with alpha*n_inst_plan >= 30 expected CAL misses (so the binomial test has power) and >= 8 configs with TUNE
    miss <= alpha - z*SE(alpha; n_inst_plan) [z=2.5 ~ 1.28 for the delta=0.10 test + ~1.2 for TUNE-to-CAL noise]; alpha_fb = ceil_to_0.05(min TUNE FDP over those
    + z * 0.5/sqrt(n_ep_plan)) (0.5 = worst-case sd of a [0,1] loss)."""
    for a in ladder:
        if a * n_inst_plan < min_expected_misses:
            continue
        S = np.where(tune_miss <= a - z * np.sqrt(a * (1 - a) / n_inst_plan))[0]
        if len(S) >= min_configs:
            hf = z * 0.5 / np.sqrt(n_ep_plan)
            return dict(miss=a, fb=float(np.round(np.ceil((tune_fb[S].min() + hf) / 0.05 - 1e-9) * 0.05, 2)), n_configs=len(S), planned_n_inst=n_inst_plan)
_t2b = choose_t2_v2(S_TUNE["miss"], S_TUNE["fb"], n_cal2_inst_planned, N_CAL2)
ALPHA_T2 = dict(miss=_t2b["miss"], fb=_t2b["fb"])                                  # THE T2 used from here on
PREREG["T2_v2"] = dict(ALPHA_T2, delta=DELTA, planned_cal2_instances=n_cal2_inst_planned, configs_within_headroom=_t2b["n_configs"],
                       fixed_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                       reason="T2-v1 gave an empty Lambda-hat on CAL1 (first path element failed); revised rule sizes alpha and headroom from LTT's own power requirement; tested on fresh CAL2")
log_event("T2-v2 rule fixed from TUNE + planned sizes only; CAL1 declared consumed")
(NB3_DIR / "prereg_targets.json").write_text(json.dumps(PREREG, indent=2))
print(f"\nT2-v2 (fixed before CAL2 is touched): alpha_miss={ALPHA_T2['miss']}, alpha_fb={ALPHA_T2['fb']}, delta={DELTA}  "
      f"({_t2b['n_configs']} configs within headroom on TUNE; planned CAL2 instances {n_cal2_inst_planned})")

# --- open CAL2 and run LTT with the SAME procedure (path from TUNE, single-start fixed sequence) ---
log_event("CAL2 split opened")
t0 = time.time()
register_work("CAL2", CORP["CAL2"], DETS["CAL2"])
_res = eval_many([("CAL2", k) for k in ALL_KEYS])
M_CAL2 = to_matrices(_res, ["CAL2"], ALL_KEYS)
IDX_CAL2 = np.arange(len(CORP["CAL2"]["episodes"]))
print(f"all {G} configs evaluated on CAL2 ({len(IDX_CAL2)} episodes, {int(M_CAL2['n_inst'].sum())} instances) in {time.time() - t0:.1f}s wall")
lam_hat2, det2 = ltt_select(M_TUNE, IDX_TUNE, M_CAL2, IDX_CAL2, ALPHA_T2, DELTA, starts=(0,))
LTT_MAIN["T2"] = dict(lam_hat=lam_hat2, **det2)
print(f"\n=== T2 (v2) on CAL2: |Lambda-hat| = {int(lam_hat2.sum())} of {G}; CAL2 p-values evaluated for {int(det2['visited'].sum())} configs (lazy walk)")
if lam_hat2.any():
    _rows = [dict(config=describe_key(ALL_KEYS[j]), cal2_miss=det2["p_test"]["miss"][j], cal2_fb=det2["p_test"]["fb"][j],
                  cal2_pooled_prec=det2["p_test"]["pooled_prec"][j], p_max=det2["p_test"]["p_max"][j]) for j in np.where(lam_hat2)[0]]
    print(pd.DataFrame(_rows).round(4).sort_values("cal2_miss").to_string(index=False))
