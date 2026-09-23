# ---- Results summary (computed from the variables above; every row labelled synthetic / measured / assumed / paper) ----
# Nothing below is typed in by hand except the two PAPER reference numbers (marked label = paper); everything else is read from the live variables of sections 3-7.
LABELS_OK = {"synthetic", "measured", "assumed", "paper"}
def _fmt(x, nd=4):
    return f"{x:.{nd}f}"

# ---- table A: the two target sets side by side (section 3; LTT, delta = DELTA) ----
def _target_row(name, alpha, key, note):
    d = LTT_MAIN[key]; lam = d["lam_hat"]; p = d["p_test"]; n_cert = int(lam.sum())
    if n_cert:
        j = J_T2                                                                             # the chosen (cheapest, ties -> fewest false blurs) member of Lambda-hat
        cfg, cal = describe_key(ALL_KEYS[j]), f"miss {_fmt(p['miss'][j])}, episode-FDP {_fmt(p['fb'][j], 3)}, pooled precision {_fmt(p['pooled_prec'][j], 3)}"
    else:
        exc = np.maximum(p["miss"] / alpha["miss"], p["fb"] / alpha["fb"]); jb = int(np.argmin(exc))
        cfg = "none (empty Lambda-hat: abstain)"
        cal = (f"closest config (NOT certified) {describe_key(ALL_KEYS[jb])}: miss {_fmt(p['miss'][jb])}, episode-FDP {_fmt(p['fb'][jb], 3)}, "
               f"pooled precision {_fmt(p['pooled_prec'][jb], 3)}; lowest miss anywhere {_fmt(p['miss'].min())}")
    return dict(target_set=name, alpha_miss=alpha["miss"], alpha_false_blur=alpha["fb"], delta=DELTA, certified=("YES" if n_cert else "NO"), lambda_hat_size=n_cert,
                chosen_config=cfg, CAL_point_estimates=cal, label="synthetic")
TARGETS_DF = pd.DataFrame([_target_row("T1: client's targets", ALPHA_T1, "T1", ""), _target_row("T2: relaxed, revised (T2-v2)", ALPHA_T2, "T2", "")])

# ---- table B: everything else, one row per number ----
ROWS = []
def add(block, item, value, label, source):
    ROWS.append(dict(block=block, item=item, value=value, label=label, source=source))

_rs = RS_STATS.set_index("procedure"); _pd = PROC_DF.set_index(["family", "procedure"])
_fine = "fine (step 0.01)"
add("LTT vs SeqCRC (re-splits)", f"LTT full grid, T2: violation fraction / abstain rate (R = {int(_rs.loc['T2', 'R'])} re-splits of the main run)",
    f"{_rs.loc['T2', 'violation_rate']:.3f} / {_rs.loc['T2', 'abstain_rate']:.3f}", "synthetic", "section 3(d); P1 Thm 1 bound: <= delta")
add("LTT vs SeqCRC (re-splits)", "LTT full grid, T1: violation fraction / abstain rate", f"{_rs.loc['T1', 'violation_rate']:.3f} / {_rs.loc['T1', 'abstain_rate']:.3f}", "synthetic", "section 3(d)")
for _p in ("CRC (instances)", "LTT-1D (miss only)", "CRC (episode-averaged)"):
    _r = _pd.loc[(_fine, _p)]
    add("LTT vs SeqCRC (re-splits)", f"{_p}, fine ladder, alpha = {ALPHA_CRC}: violation fraction / abstain rate / mean true miss (R = {int(_r['R'])})",
        f"{_r['violation_rate']:.3f} / {_r['abstain_rate']:.3f} / {_r['mean_true_miss']:.4f}", "synthetic", "section 4; CRC controls the mean, LTT the probability")
_sn = SWEEP_N_DF.set_index(["cal_episodes", "procedure"])
add("LTT vs SeqCRC (re-splits)", f"smallest calibration set ({int(SWEEP_N[0])} episodes): CRC violation vs LTT-1D violation (LTT-1D abstain)",
    f"{_sn.loc[(SWEEP_N[0], 'CRC'), 'violation_rate']:.3f} vs {_sn.loc[(SWEEP_N[0], 'LTT-1D'), 'violation_rate']:.3f} ({_sn.loc[(SWEEP_N[0], 'LTT-1D'), 'abstain_rate']:.2f})", "synthetic", "section 4 size sweep")

for _nm in ("T2", "BEST"):
    _a = AUD[_nm]
    add("Sealed audit (delta = %.2f)" % DELTA_AUDIT, f"Prop. 26 on {_nm} config {describe_key(_a['key'])}: C/Q = {_a['C']}/{_a['Q']} -> miss-rate bound U_Q(C, delta) / recall lower bound",
        f"{_a['U']:.4f} / {1 - _a['U']:.4f}", "synthetic", "P3 Prop. 26 (Clopper-Pearson over audited face instances); the ONLY recall basis (Remark 9)")
add("Sealed audit (delta = %.2f)" % DELTA_AUDIT, f"excluded-pool missed mass (T2 config): K0 = {K0}/{N0_SAMPLE}; Thm 8 r <= M_U/N ; Thm 7 r <= p0*U",
    f"{r_hyper:.4f} ; {r_binom:.4f}", "synthetic", "P3 Thm 7/8 on 45-s segments (segment length derived here); NOT a recall claim")
add("Sealed audit (delta = %.2f)" % DELTA_AUDIT, f"implied SEGMENT-level recall lower bound (prevalence lower bound a = {a_pi:.3f}, joint confidence {1 - DELTA_AUDIT:.2f})",
    f"{recall_seg_lb:.3f}", "synthetic", "P3 Remark 9: recall >= 1 - r/a")
add("Sealed audit (delta = %.2f)" % DELTA_AUDIT, "Monte Carlo coverage of the excluded-pool bounds: Thm 8 finite pool / Thm 7 on dependent segments", f"{cov_thm8:.4f} / {cov_thm7:.4f}", "synthetic", "section 5(b); nominal 0.975 each")
add("Sealed audit (delta = %.2f)" % DELTA_AUDIT, "Thm 19 walk returns threshold: segment-mass version (eps = %.2f) / instance-level twin (eps = %.2f, derived here)" % (EPS_SEG, EPS_INST),
    f"{NESTED_THR[ret_seg] if ret_seg >= 0 else 'none'} / {NESTED_THR[ret_inst] if ret_inst >= 0 else 'none'}", "synthetic", "P3 Thm 19; nesting violations on the audit: %d instance-level, %d pool-level (0 = exactly nested)" % (inst_viol, pool_viol))
for _, _t in T29.iterrows():
    add("Sealed audit (delta = %.2f)" % DELTA_AUDIT, f"Thm 29 stress bounds, {_t['config']} config (declared generator only): rho_max / rho_avg at delta/(2|G|)", f"{_t['rho_max_bound']:.4f} / {_t['rho_avg_bound']:.4f}", "synthetic", "P3 Thm 29; valid only for the declared perturbations")
for _, _p in PREC_DF.iterrows():
    add("Sealed audit (delta = %.2f)" % DELTA_AUDIT, f"client's POOLED precision, {_p['config']} config: point and episode-cluster bootstrap 95% interval", f"{_p['pooled_precision']:.3f} {_p['boot95']}", "synthetic", "bootstrap, NOT a P1/P2/P3 guarantee")

add("Cost", "decode / YuNet 4K detect / libx264 encode throughput on this shared WSL2 VM (fps; several-fold run-to-run variance)",
    f"{THROUGHPUT['decode_fps']:.1f} / {THROUGHPUT['detect_fps']:.2f} / {THROUGHPUT['encode_fps']:.2f}", "measured", "data/smoke/timings.json via section 3 throughput dict")
add("Cost", "CPU / GPU price per hour; GPU speed-up", f"${PRICE['assumed_cpu_usd_per_hour']} / ${PRICE['assumed_gpu_usd_per_hour']} ; {THROUGHPUT['assumed_gpu_speedup']:.0f}x", "assumed", "placeholders, not provider quotes")
for _s in sorted(COST_BY_STRIDE):
    add("Cost", f"$/video-hour at stride {_s}: CPU (measured throughput x assumed price) / GPU (assumed speed-up x assumed price)",
        f"{COST_BY_STRIDE[_s]['cpu_usd_per_video_hour']:.3f} / {COST_BY_STRIDE[_s]['gpu_usd_per_video_hour']:.3f}", "assumed", "processing_cost; only stride 1 is certified here (all Lambda-hat members are stride 1)")
add("Cost", "GPU break-even speed-up = GPU price / CPU price (GPU cheaper only above it)", f"{BE_SPEEDUP:.1f}x", "assumed", "section 6 identity")
_sv = VAR_DF[VAR_DF["stride"] == 1]["cpu_usd_per_video_hour"]
add("Cost", "CPU $/video-hour at stride 1 across the recorded slow / measured / fast smoke runs", " / ".join(f"{v:.2f}" for v in _sv), "assumed", "throughput ranges recorded in Notebook 2 (measured) x the ASSUMED price: the dollar figure is assumed")

_best_recall_ok = any(AUD[n]["U"] <= EPS_TARGET for n in AUD)
_best_prec_ok = any(AUD[n]["prec_ci"][0] >= 0.95 for n in AUD)
add("Client's claims", "99% recall (miss rate <= 1%) certified for ANY audited config? (needs U_Q(C, delta) <= 0.01)", "YES" if _best_recall_ok else "NO", "synthetic",
    "; ".join(f"{n}: U = {AUD[n]['U']:.4f}" for n in AUD))
add("Client's claims", "95% pooled precision supported for ANY audited config? (bootstrap lower limit >= 0.95; a bootstrap, not a guarantee)", "YES" if _best_prec_ok else "NO", "synthetic",
    "; ".join(f"{n}: lower limit {AUD[n]['prec_ci'][0]:.3f}" for n in AUD))
_nest1 = T19.loc[T19["U_Q_C_inst"] <= EPS_TARGET, "thr"].tolist()                                          # single-config Prop. 26 at delta (post hoc: eps = 1% was NOT pre-registered for this family)
_nestM = [k[0] for k in NESTED_KEYS if float(cp_upper(int(NEST[k]["missed"].sum()), len(NEST[k]["missed"]), DELTA_AUDIT / len(NESTED_KEYS))) <= EPS_TARGET]
add("Client's claims", f"POST-HOC look at the pre-registered nested family: thresholds with U_Q(C, delta) <= 1% at delta / with a delta/M union bound over the M = {len(NESTED_KEYS)} members (pooled precision of those at delta)",
    f"{_nest1} / {_nestM if _nestM else 'none'} (precision {T19.loc[T19['U_Q_C_inst'] <= EPS_TARGET, 'pooled_precision'].tolist()})", "synthetic",
    f"eps = 1% was not pre-registered for this family (registered eps_inst = {EPS_INST}): NOT a certificate of the 99% claim; and the 95% precision claim fails for every member")
add("Client's claims", "T1 (client targets) certified by LTT on CAL?", TARGETS_DF.iloc[0]["certified"], "synthetic", "section 3")
add("Re-verification harness", "verdicts (LTT targets | client 99%/95%) for cases i, ii, iii-a, iii-b, iv", "; ".join(f"{v['verdict_ltt_targets']}|{v['verdict_client_claims']}" for v in H.values()), "synthetic", "section 7")
add("Paper reference", "P3 Sec. 8: U_300(0, 0.025) = 0.01222 (paper) vs this notebook's Clopper-Pearson", f"0.01222 (paper) vs {float(cp_upper(0, 300, 0.025)):.5f}", "paper", "P3 Sec. 8 worked example [facered-q8]")
add("Paper reference", "P3 Remark 9: pi = 0.01 and r <= 0.005 give recall >= (paper) 0.5", f"{1 - 0.005 / 0.01:.1f}", "paper", "[facered-q9, facered-q11]")
SUMMARY_DF = pd.DataFrame(ROWS)

pd.set_option("display.width", 320); pd.set_option("display.max_colwidth", 150); pd.set_option("display.max_rows", 200)
print("A. The two target sets side by side (LTT, delta = %.2f; CAL/CAL2 point estimates, synthetic data)" % DELTA)
print(TARGETS_DF.to_string(index=False))
print("\nB. Everything else, one row per number")
print(SUMMARY_DF.to_string(index=False))
_last = {c: v for c, _d, v in CHECKS}                                              # last verdict per check id
_not_pass = [c for c, v in _last.items() if v != "PASS"]
print(f"\nSelf-checks so far: {len(_last)} distinct ids, {len(_last) - len(_not_pass)} PASS; NOT PASS: {_not_pass}  (each is reported in the section that ran it)")

check("R.1", "every row of both summary tables carries a label from {synthetic, measured, assumed, paper} (a-priori: no unlabelled number)",
      bool(TARGETS_DF["label"].isin(LABELS_OK).all() and SUMMARY_DF["label"].isin(LABELS_OK).all()), f"{len(TARGETS_DF)} + {len(SUMMARY_DF)} rows; labels used: {sorted(set(TARGETS_DF['label']) | set(SUMMARY_DF['label']))}")
check("R.2", "the summary's verdict on the client's 99% recall claim equals an independent recomputation from the audit bounds (Prop. 26 U_Q(C, delta) <= 0.01) and from section 5's own table",
      _best_recall_ok == bool(P26_DF["claims_recall_99"].any()) == any(float(cp_upper(AUD[n]["C"], AUD[n]["Q"], DELTA_AUDIT)) <= EPS_TARGET for n in AUD),
      f"summary says {'YES' if _best_recall_ok else 'NO'}; section 5 table claims_recall_99 = {P26_DF['claims_recall_99'].tolist()}")
check("R.3", "the target table is consistent with the audit: the T2 row's chosen config is the config that was pre-registered and audited, and 'certified = NO' for T1 iff Lambda-hat is empty",
      TARGETS_DF.iloc[1]["chosen_config"] == describe_key(AUD["T2"]["key"]) and (TARGETS_DF.iloc[0]["certified"] == "NO") == (int(LTT_MAIN["T1"]["lam_hat"].sum()) == 0),
      f"T1 |Lambda-hat| = {int(LTT_MAIN['T1']['lam_hat'].sum())}; T2 chosen {TARGETS_DF.iloc[1]['chosen_config']}")
