# ---- 3(d) Verification by simulation: R random re-splits of a pool; "true" risks from the separate TRUTH population ----
def concat_M(Ms):
    return dict(n_inst=np.concatenate([m["n_inst"] for m in Ms]), **{f: np.concatenate([m[f] for m in Ms], axis=1) for f in ("n_miss", "n_box", "n_false", "fdp")})
M_POOL = concat_M([M_TUNE, M_CAL, M_CAL2])                       # 650 + 950 + 950 = 2,550 episodes, all 58 configs
E_POOL = len(M_POOL["n_inst"]); R_SPLITS = 200
print(f"re-split pool: {E_POOL} episodes ({int(M_POOL['n_inst'].sum())} instances); each re-split draws TUNE={N_TUNE} and CAL={N_CAL} episodes at random (same sizes as the main run)")

# ---- TRUTH: lazily evaluated only for the configs some procedure actually selects (cache) ----
for _n, _c in TRUTH_CH.items():
    register_work(_n, _c, TRUTH_CH_DET[_n])
TRUTH_CACHE = {}
def truth_risks(keys):
    missing = [k for k in dict.fromkeys(keys) if k not in TRUTH_CACHE]
    if missing:
        res = eval_many([(n, k) for k in missing for n in TRUTH_CH])
        for k in missing:
            M = to_matrices(res, list(TRUTH_CH), [k])
            ni = M["n_inst"].sum()
            TRUTH_CACHE[k] = dict(miss=float(M["n_miss"][0].sum() / ni), fb=float(M["fdp"][0].mean()), n_inst=int(ni), n_ep=len(M["n_inst"]),
                                  pooled_prec=float(1 - M["n_false"][0].sum() / max(M["n_box"][0].sum(), 1)))
    return {k: TRUTH_CACHE[k] for k in keys}

def pick_cheapest(lam_hat, p_test):
    """Among Lambda-hat: minimum $/video-hour, then minimum CAL episode-FDP, then minimum CAL miss (any lambda in Lambda-hat keeps the guarantee, P1 Sec. 2.1)."""
    idx = np.where(lam_hat)[0]
    if len(idx) == 0:
        return -1
    order = np.lexsort((p_test["miss"][idx], p_test["fb"][idx], np.round(COST_VEC[idx], 9)))
    return int(idx[order[0]])

# target sets and start sets simulated: T1, T2 (single start), T2 with multi-start J=3 and J=5 (each start tested at delta/|J|), and T2-tight: the boundary case
# alpha = 1.4 x the TRUE risks of the main run's chosen T2 config, so that some grid configs are truly (slightly) unsafe and the guarantee is not vacuous.
# (A first attempt with factor 1.0 abstained on every one of 200 re-splits -- LTT needs headroom -- so the factor was loosened to 1.4 AFTER seeing that; this is a
#  demonstration knob, not a claimed threshold.)
_jm = pick_cheapest(LTT_MAIN["T2"]["lam_hat"], LTT_MAIN["T2"]["p_test"])
_tm = truth_risks([ALL_KEYS[_jm]])[ALL_KEYS[_jm]]
ALPHA_TIGHT = dict(miss=round(1.4 * _tm["miss"], 4), fb=round(min(1.4 * _tm["fb"], 0.95), 4))
PROCS = {"T1": (ALPHA_T1, (0,)), "T2": (ALPHA_T2, (0,)), "T2 multi-start J=3": (ALPHA_T2, (0, 15, 30)), "T2 multi-start J=5": (ALPHA_T2, (0, 12, 24, 36, 48)),
         "T2-tight (alpha = 1.4 x true risk of the main-run choice)": (ALPHA_TIGHT, (0,))}
t0 = time.time()
rng_rs = np.random.default_rng(SEED + 11)
RS = {name: dict(chosen=[], size=[], visited=[]) for name in PROCS}
for r in range(R_SPLITS):
    perm = rng_rs.permutation(E_POOL)
    tune_i, cal_i = perm[:N_TUNE], perm[N_TUNE:N_TUNE + N_CAL]
    for name, (alpha, starts) in PROCS.items():
        lam, det = ltt_select(M_POOL, tune_i, M_POOL, cal_i, alpha, DELTA, starts=starts)
        RS[name]["chosen"].append(pick_cheapest(lam, det["p_test"])); RS[name]["size"].append(int(lam.sum())); RS[name]["visited"].append(int(det["visited"].sum()))
t_rs = time.time() - t0
for name in RS:
    RS[name] = {k: np.array(v) for k, v in RS[name].items()}
print(f"{R_SPLITS} re-splits x {len(PROCS)} procedures in {t_rs:.1f}s; distinct configs ever selected: "
      + ", ".join(f"{n[:18]} {len(set(RS[n]['chosen'][RS[n]['chosen'] >= 0]))}" for n in RS))
t0 = time.time()
_need = [ALL_KEYS[j] for name in RS for j in set(RS[name]["chosen"][RS[name]["chosen"] >= 0].tolist())]
_ = truth_risks(_need)
print(f"TRUTH ({N_TRUTH} episodes, {TRUTH_CACHE[_need[0]]['n_inst']} instances) evaluated for {len(TRUTH_CACHE)} selected configs in {time.time() - t0:.1f}s")

def rs_stats(name, alpha):
    ch = RS[name]["chosen"]; sel = ch >= 0
    tr = [TRUTH_CACHE[ALL_KEYS[j]] for j in ch[sel]]
    viol = np.array([(t["miss"] > alpha["miss"]) or (t["fb"] > alpha["fb"]) for t in tr], bool)
    return dict(procedure=name, alpha_miss=alpha["miss"], alpha_fb=alpha["fb"], R=len(ch), abstain_rate=float(1 - sel.mean()), violations=int(viol.sum()),
                violation_rate=float(viol.sum() / len(ch)), mean_lambda_hat_size=float(RS[name]["size"].mean()),
                mean_cal_configs_evaluated=float(RS[name]["visited"].mean()),
                worst_true_miss=max([t["miss"] for t in tr], default=float("nan")), worst_true_fdp=max([t["fb"] for t in tr], default=float("nan")))
RS_STATS = pd.DataFrame([rs_stats(n, PROCS[n][0]) for n in PROCS])
pd.set_option("display.width", 260)
print(); print(RS_STATS.round(4).to_string(index=False))
tol = 3 * np.sqrt(DELTA * (1 - DELTA) / R_SPLITS)
_st = RS_STATS.set_index("procedure")
for cid, name in (("3.2", "T1"), ("3.3", "T2"), ("3.5", "T2-tight (alpha = 1.4 x true risk of the main-run choice)")):
    r = _st.loc[name]
    check(cid, f"{name}: over {R_SPLITS} re-splits the chosen config's TRUE risk exceeds alpha on either risk in at most delta + 3 binomial SE = {DELTA + tol:.3f} of the runs "
          f"(P1 Thm 1; a-priori){' -- VACUOUS when every split abstains: nothing is selected' if name == 'T1' else ''}", r["violation_rate"] <= DELTA + tol, f"violation {int(r['violations'])}/{R_SPLITS} = {r['violation_rate']:.3f}; abstain rate {r['abstain_rate']:.3f}")
_ab = [_st.loc[n]["abstain_rate"] for n in ("T2", "T2 multi-start J=3", "T2 multi-start J=5")]
check("3.6", "multi-start costs power: abstain rate is non-decreasing in |J| (J=1 -> 3 -> 5; each start tested at delta/|J|) (a-priori)", _ab[0] <= _ab[1] + 1e-9 <= _ab[2] + 2e-9,
      f"abstain {np.round(_ab, 3).tolist()}")
