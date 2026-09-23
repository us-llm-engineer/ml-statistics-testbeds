# ---- 4. SeqCRC-style comparison. ONE-parameter CRC (P2 Eq. 1) on the confidence threshold; the two-step localisation stage is NOT implemented (budget; see text). ----
# Two threshold ladders (fixed a priori, not learned; everything else fixed at gap=30, pad=20, margin=0.3, stride=1):
#   'coarse' = 0.600, 0.625, ..., 0.850 (11 values)      'fine' = 0.70, 0.71, ..., 0.80 (11 values, around where the miss curve crosses alpha)
FAM_DEF = {"coarse (step 0.025)": [round(0.6 + 0.025 * i, 3) for i in range(11)], "fine (step 0.01)": [round(0.70 + 0.01 * i, 3) for i in range(11)]}
FAM_KEYS_BY = {n: [(t, 1, 30, 20, 0.3) for t in thr] for n, thr in FAM_DEF.items()}
NEW_FAM = sorted({k for ks in FAM_KEYS_BY.values() for k in ks if k not in KEY_INDEX})
t0 = time.time()
_res = eval_many([(s, k) for s in ("CAL", "CAL2") for k in NEW_FAM])
_MN = {s: to_matrices({(s, k): _res[(s, k)] for k in NEW_FAM}, [s], NEW_FAM) for s in ("CAL", "CAL2")}
def fam_matrix(s, M_grid, keys):
    out = {f: np.stack([M_grid[f][KEY_INDEX[k]] if k in KEY_INDEX else _MN[s][f][NEW_FAM.index(k)] for k in keys]) for f in ("n_miss", "n_box", "n_false", "fdp")}
    out["n_inst"] = M_grid["n_inst"]
    return out
M_FAM_BY = {n: concat_M([fam_matrix("CAL", M_CAL, ks), fam_matrix("CAL2", M_CAL2, ks)]) for n, ks in FAM_KEYS_BY.items()}      # [11 thresholds, 1,900 episodes]
E_FAM = len(M_FAM_BY["fine (step 0.01)"]["n_inst"])
print(f"two threshold families ({len(NEW_FAM)} newly evaluated configs) on the {E_FAM} CAL+CAL2 episodes in {time.time() - t0:.1f}s")

ALPHA_CRC = ALPHA_T2["miss"]                     # same miss target as T2 (a-priori)
def crc_select(M, idx, alpha, B=1.0, level="instance"):
    """CRC Eq. 1 [P2, facered-q4]: lambda-hat = inf{lambda : (1/(n+1)) sum_i L_i(lambda) + B/(n+1) <= alpha}, here lambda = 1 - thr, so the
    LARGEST thr satisfying it. Risk curve is made monotone in thr by the least non-decreasing majorant (P2 Sec. IV-E monotonization idea). Returns
    the family index; if no threshold satisfies the constraint the most conservative one (index 0) is used (convention inf(empty) = max(lambda))."""
    if level == "instance":
        n = M["n_inst"][idx].sum(); L_sum = M["n_miss"][:, idx].sum(1).astype(float)          # unit: face INSTANCE (derived here)
    else:   # episode-averaged image-style loss: episodes without faces contribute 0 (P2 Sec. IV-E edge case), so the average is DILUTED
        ni = M["n_inst"][idx]; n = len(idx)
        L_sum = (np.where(ni > 0, M["n_miss"][:, idx] / np.maximum(ni, 1), 0.0)).sum(1)
    risk = np.maximum.accumulate(L_sum)                                                       # monotone non-decreasing in thr
    ok = (risk + B) / (n + 1) <= alpha
    return int(np.where(ok)[0].max()) if ok.any() else 0

def ltt1d_select(M, idx, alpha, delta=DELTA):
    """LTT fixed sequence, MISS risk only, along the natural path (lowest threshold = safest first); exact binomial p-values, level delta.
    Returns the largest accepted threshold index (fewest false blurs), or -1 = abstain."""
    n = M["n_inst"][idx].sum(); k = M["n_miss"][:, idx].sum(1)
    acc, _ = fixed_sequence(p_binom(k, n, alpha), delta, starts=(0,))
    return int(np.where(acc)[0].max()) if acc.any() else -1

R_CRC = 300
PROC_BY, PROC_DFS = {}, []
for fam_name, keys in FAM_KEYS_BY.items():
    Mf = M_FAM_BY[fam_name]
    rng_c = np.random.default_rng(SEED + 12)
    P = {name: [] for name in ("CRC (instances)", "LTT-1D (miss only)", "CRC (episode-averaged)")}
    for r in range(R_CRC):
        idx = rng_c.permutation(E_FAM)[:N_CAL]
        P["CRC (instances)"].append(crc_select(Mf, idx, ALPHA_CRC)); P["LTT-1D (miss only)"].append(ltt1d_select(Mf, idx, ALPHA_CRC))
        P["CRC (episode-averaged)"].append(crc_select(Mf, idx, ALPHA_CRC, level="episode"))
    PROC_BY[fam_name] = {k: np.array(v) for k, v in P.items()}
    _ = truth_risks([keys[j] for v in PROC_BY[fam_name].values() for j in set(v[v >= 0].tolist())])
print(f"TRUTH risks known for {len(TRUTH_CACHE)} configs")

def proc_stats(fam_name, name, sel_idx, alpha_miss=ALPHA_CRC, alpha_fb=ALPHA_T2["fb"]):
    keys = FAM_KEYS_BY[fam_name]; ok = sel_idx >= 0
    tr = [TRUTH_CACHE[keys[j]] for j in sel_idx[ok]]
    miss = np.array([t["miss"] for t in tr]); fb = np.array([t["fb"] for t in tr]); pp = np.array([t["pooled_prec"] for t in tr])
    n_viol = int((miss > alpha_miss).sum())
    return dict(family=fam_name, procedure=name, R=len(sel_idx), abstain_rate=float(1 - ok.mean()), violations_miss=n_viol, violation_rate=n_viol / len(sel_idx),
                mean_true_miss=float(miss.mean()) if len(miss) else float("nan"), sd_true_miss=float(miss.std()) if len(miss) else float("nan"),
                mean_true_fdp=float(fb.mean()) if len(fb) else float("nan"), mean_true_pooled_precision=float(pp.mean()) if len(pp) else float("nan"),
                frac_runs_fdp_above_T2_alpha_fb=float((fb > alpha_fb).sum() / len(sel_idx)))
PROC_DF = pd.DataFrame([proc_stats(f, k, v) for f, P in PROC_BY.items() for k, v in P.items()])
# LTT-full (both risks, T2) from section 3(d): the same alpha_miss, evaluated on the miss target only for a like-for-like column
_ch = RS["T2"]["chosen"]; _sel = _ch >= 0
_tr = [TRUTH_CACHE[ALL_KEYS[j]] for j in _ch[_sel]]
LTT_FULL = dict(family="full grid", procedure="LTT-full T2 (both risks; section 3)", R=len(_ch), abstain_rate=float(1 - _sel.mean()),
                violations_miss=int(sum(t["miss"] > ALPHA_T2["miss"] for t in _tr)), violation_rate=float(sum(t["miss"] > ALPHA_T2["miss"] for t in _tr) / len(_ch)),
                mean_true_miss=float(np.mean([t["miss"] for t in _tr])), sd_true_miss=float(np.std([t["miss"] for t in _tr])),
                mean_true_fdp=float(np.mean([t["fb"] for t in _tr])), mean_true_pooled_precision=float(np.mean([t["pooled_prec"] for t in _tr])),
                frac_runs_fdp_above_T2_alpha_fb=float(sum(t["fb"] > ALPHA_T2["fb"] for t in _tr) / len(_ch)))
PROC_DF = pd.concat([PROC_DF, pd.DataFrame([LTT_FULL])], ignore_index=True)
pd.set_option("display.width", 260)
print(f"\nmiss target alpha = {ALPHA_CRC} for every row; R = {R_CRC} random {N_CAL}-episode calibration sets from the {E_FAM}-episode CAL+CAL2 pool; 'true' risks on the {N_TRUTH}-episode TRUTH population")
print(PROC_DF.round(4).to_string(index=False))
print("TRUE miss risk along each ladder (TRUTH):", {f: [round(TRUTH_CACHE[k]["miss"], 3) for k in ks if k in TRUTH_CACHE] for f, ks in FAM_KEYS_BY.items()})

tolc = 3 * np.sqrt(DELTA * (1 - DELTA) / R_CRC)
def row(fam, proc):
    return PROC_DF[(PROC_DF["family"] == fam) & (PROC_DF["procedure"] == proc)].iloc[0]
for cid, fam in (("4.1a", "coarse (step 0.025)"), ("4.1b", "fine (step 0.01)")):
    r = row(fam, "CRC (instances)"); se = r["sd_true_miss"] / np.sqrt(R_CRC)
    check(cid, f"CRC on the {fam} ladder (instance loss, P2 Thm 1): MEAN true miss of the selected config <= alpha + 3 MC SE (expectation control holds)",
          r["mean_true_miss"] <= ALPHA_CRC + 3 * se, f"mean {r['mean_true_miss']:.4f} vs alpha {ALPHA_CRC} (MC SE {se:.4f})")
for cid, fam in (("4.2a", "coarse (step 0.025)"), ("4.2b", "fine (step 0.01)")):
    r = row(fam, "CRC (instances)")
    check(cid, f"CRC on the {fam} ladder: per-split violation fraction P(true miss > alpha) EXCEEDS delta + 3 SE = {DELTA + tolc:.3f} (expectation control is not a 1-delta guarantee; "
          f"a-priori claim; the coarse ladder was run first)", r["violation_rate"] > DELTA + tolc, f"{int(r['violations_miss'])}/{R_CRC} = {r['violation_rate']:.3f}; mean true miss {r['mean_true_miss']:.4f}")
for cid, fam in (("4.3a", "coarse (step 0.025)"), ("4.3b", "fine (step 0.01)")):
    r = row(fam, "LTT-1D (miss only)")
    check(cid, f"LTT-1D on the {fam} ladder (same alpha, delta=0.10): per-split violation fraction <= delta + 3 SE (P1 Thm 1)", r["violation_rate"] <= DELTA + tolc,
          f"{int(r['violations_miss'])}/{R_CRC} = {r['violation_rate']:.3f}; abstain {r['abstain_rate']:.3f}")
r = row("fine (step 0.01)", "CRC (episode-averaged)")
check("4.4", "face-free episodes dilute an episode-averaged CRC (fine ladder): the instance-level true miss of its selected config exceeds alpha on average by >= 30% (a-priori; contract rule 7)",
      r["mean_true_miss"] >= 1.3 * ALPHA_CRC, f"mean instance-level miss {r['mean_true_miss']:.4f} vs alpha {ALPHA_CRC}; violation {r['violation_rate']:.3f}")

# ---- alpha sweep on the fine ladder (a-priori grid of targets): where does CRC's expectation control leave the per-split violation? ----
SWEEP_ALPHAS = [0.03, 0.04, 0.05, 0.06, 0.07]
fine = "fine (step 0.01)"; fkeys = FAM_KEYS_BY[fine]; Mf = M_FAM_BY[fine]
_ = truth_risks(fkeys)                                              # the whole fine ladder on TRUTH (a few extra configs)
SWEEP = []
for a in SWEEP_ALPHAS:
    rng_c = np.random.default_rng(SEED + 13)
    sel_c, sel_l = [], []
    for r in range(R_CRC):
        idx = rng_c.permutation(E_FAM)[:N_CAL]
        sel_c.append(crc_select(Mf, idx, a)); sel_l.append(ltt1d_select(Mf, idx, a))
    for proc, sel in (("CRC", np.array(sel_c)), ("LTT-1D", np.array(sel_l))):
        ok = sel >= 0
        tm = np.array([TRUTH_CACHE[fkeys[j]]["miss"] for j in sel[ok]]); tf = np.array([TRUTH_CACHE[fkeys[j]]["pooled_prec"] for j in sel[ok]])
        SWEEP.append(dict(alpha=a, procedure=proc, abstain_rate=float(1 - ok.mean()), violation_rate=float((tm > a).sum() / R_CRC),
                          mean_true_miss=float(tm.mean()) if len(tm) else float("nan"), mean_true_pooled_precision=float(tf.mean()) if len(tf) else float("nan")))
SWEEP_DF = pd.DataFrame(SWEEP)
print("\nalpha sweep (fine ladder, same 300 calibration sets per alpha):"); print(SWEEP_DF.round(4).to_string(index=False))
_c, _l = SWEEP_DF[SWEEP_DF["procedure"] == "CRC"], SWEEP_DF[SWEEP_DF["procedure"] == "LTT-1D"]
check("4.5", f"over the alpha sweep, CRC's mean true miss stays <= alpha + 3 MC SE at EVERY alpha (expectation control) while its per-split violation exceeds delta + 3 SE = {DELTA + tolc:.3f} "
      "at AT LEAST ONE alpha (a-priori claim; the sweep is the honest way to see the gap between 'in expectation' and 'with probability 1-delta')",
      bool((_c["mean_true_miss"] <= _c["alpha"] + 0.002).all() and (_c["violation_rate"] > DELTA + tolc).any()),
      f"CRC violation by alpha: {dict(zip(_c['alpha'], _c['violation_rate'].round(3)))}; mean true miss {dict(zip(_c['alpha'], _c['mean_true_miss'].round(4)))}")
check("4.6", f"over the alpha sweep, LTT-1D's per-split violation is <= delta + 3 SE = {DELTA + tolc:.3f} at EVERY alpha (P1 Thm 1; a-priori)", bool((_l["violation_rate"] <= DELTA + tolc).all()),
      f"LTT-1D violation by alpha: {dict(zip(_l['alpha'], _l['violation_rate'].round(3)))}; abstain {dict(zip(_l['alpha'], _l['abstain_rate'].round(2)))}")

# ---- calibration-size sweep at alpha = 0.05: the smaller the calibration set, the more R(lambda-hat) disperses around its mean ----
SWEEP_N = [100, 200, 400, 950]                                       # calibration episodes (~0.67 face instances per episode)
SWEEP_N_ROWS = []
for n_ep in SWEEP_N:
    rng_c = np.random.default_rng(SEED + 14)
    sel_c, sel_l = [], []
    for r in range(R_CRC):
        idx = rng_c.permutation(E_FAM)[:n_ep]
        sel_c.append(crc_select(Mf, idx, ALPHA_CRC)); sel_l.append(ltt1d_select(Mf, idx, ALPHA_CRC))
    for proc, sel in (("CRC", np.array(sel_c)), ("LTT-1D", np.array(sel_l))):
        ok = sel >= 0
        tm = np.array([TRUTH_CACHE[fkeys[j]]["miss"] for j in sel[ok]])
        SWEEP_N_ROWS.append(dict(cal_episodes=n_ep, cal_instances_approx=int(round(n_ep * Mf["n_inst"].sum() / E_FAM)), procedure=proc, abstain_rate=float(1 - ok.mean()),
                                 violation_rate=float((tm > ALPHA_CRC).sum() / R_CRC), mean_true_miss=float(tm.mean()) if len(tm) else float("nan"),
                                 sd_true_miss=float(tm.std()) if len(tm) else float("nan")))
SWEEP_N_DF = pd.DataFrame(SWEEP_N_ROWS)
print(f"\ncalibration-size sweep (fine ladder, alpha = {ALPHA_CRC}):"); print(SWEEP_N_DF.round(4).to_string(index=False))
_cn, _ln = SWEEP_N_DF[SWEEP_N_DF["procedure"] == "CRC"].set_index("cal_episodes"), SWEEP_N_DF[SWEEP_N_DF["procedure"] == "LTT-1D"].set_index("cal_episodes")
check("4.7", f"with a SMALL calibration set ({SWEEP_N[0]} episodes) CRC's per-split violation exceeds delta + 3 SE = {DELTA + tolc:.3f} while LTT-1D's stays below it "
      "(a-priori claim: expectation control disperses with small n; LTT pays with abstention instead)",
      bool(_cn.loc[SWEEP_N[0], "violation_rate"] > DELTA + tolc and _ln.loc[SWEEP_N[0], "violation_rate"] <= DELTA + tolc),
      f"CRC violation by n: {_cn['violation_rate'].round(3).to_dict()}; LTT-1D: {_ln['violation_rate'].round(3).to_dict()}; LTT-1D abstain: {_ln['abstain_rate'].round(2).to_dict()}")
