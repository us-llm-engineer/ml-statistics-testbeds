# ---- Clopper-Pearson machinery [P3 Sec. 4, facered-q7]; U_n(k, a) = sup{q : P(Bin(n,q) <= k) >= a}, a = FAILURE probability ----
def cp_upper(k, n, a):
    k, n = np.asarray(k, float), np.asarray(n, float)
    return np.where(k >= n, 1.0, stats.beta.ppf(1 - a, k + 1, np.maximum(n - k, 1e-12)))

def cp_lower(k, n, a):
    k, n = np.asarray(k, float), np.asarray(n, float)
    return np.where(k <= 0, 0.0, stats.beta.ppf(a, np.maximum(k, 1e-12), np.maximum(n - k + 1, 1e-12)))

def min_q(C: int, eps: float, delta: float, q_max: int = 20000) -> int:
    """Smallest number of audited face instances Q with U_Q(C, delta) <= eps (Prop. 26 sizing)."""
    q = np.arange(max(C, 1), q_max + 1)
    ok = cp_upper(C, q, delta) <= eps
    return int(q[np.argmax(ok)]) if ok.any() else -1

# ---- (a) PAPER numbers: P3 Sec. 4.6 planning table, delta = 0.05 ----
PAPER_PLAN = {0.10: (29, 51), 0.05: (59, 104), 0.02: (149, 263), 0.01: (299, 528), 0.005: (598, 1058)}
plan_rows = []
for eps, (p_single, p_ten) in PAPER_PLAN.items():
    n_single = int(np.ceil(np.log(1 / 0.05) / np.log(1 / (1 - eps))))
    n_ten = int(np.ceil(np.log(10 / 0.05) / np.log(1 / (1 - eps))))
    plan_rows.append(dict(eps=eps, paper_single=p_single, computed_single=n_single, paper_10_prefixes=p_ten, computed_10_prefixes=n_ten))
PLAN_DF = pd.DataFrame(plan_rows)
print("PAPER numbers (P3 Sec. 4.6, delta=0.05, zero observed misses) vs the formula n0 >= log(1/delta)/log(1/(1-eps)):")
print(PLAN_DF.to_string(index=False))
check("2.1", "the P3 Sec. 4.6 planning table (10 printed numbers) is reproduced exactly by n0 >= log(M/delta)/log(1/(1-eps))",
      bool((PLAN_DF["paper_single"] == PLAN_DF["computed_single"]).all() and (PLAN_DF["paper_10_prefixes"] == PLAN_DF["computed_10_prefixes"]).all()),
      f"single {PLAN_DF['computed_single'].tolist()}, 10 prefixes {PLAN_DF['computed_10_prefixes'].tolist()}")

# ---- (b) DERIVED here from Prop. 26: audited FACE INSTANCES Q needed for L <= 1% with C misses allowed ----
EPS_TARGET = 0.01
sizing_rows = []
for C in [0, 1, 2, 3, 5]:
    row = dict(allowed_misses_C=C)
    for delta in (0.05, 0.10):
        row[f"Q_min_delta{delta}"] = min_q(C, EPS_TARGET, delta)
    sizing_rows.append(row)
SIZING_DF = pd.DataFrame(sizing_rows)
print("\nProp. 26 sizing (computed here, not hard-coded): audited face instances Q with U_Q(C, delta) <= 0.01")
print(SIZING_DF.to_string(index=False))
_q0 = SIZING_DF.loc[SIZING_DF["allowed_misses_C"] == 0, "Q_min_delta0.05"].iloc[0]
_defs_ok = all(cp_upper(C, q, 0.05) <= EPS_TARGET < cp_upper(C, q - 1, 0.05)
               for C, q in zip(SIZING_DF["allowed_misses_C"], SIZING_DF["Q_min_delta0.05"]))
check("2.2", "Prop. 26 sizing: C=0 needs 299 (the same number as P3 Sec. 4.6 for eps=0.01), Q_min increases with C, and each Q_min is the exact "
      "threshold (U_Q <= 0.01 < U_{Q-1})",
      _q0 == 299 and SIZING_DF["Q_min_delta0.05"].is_monotonic_increasing and _defs_ok,
      f"Q_min(C=0,1,2,3,5) = {SIZING_DF['Q_min_delta0.05'].tolist()} at delta=0.05; {SIZING_DF['Q_min_delta0.1'].tolist()} at delta=0.10")
Q_AUDIT = int(CORP["AUDIT"]["tracks"]["is_face_to_redact"].sum())
C_MAX_AUDIT = max([C for C in range(0, 12) if cp_upper(C, Q_AUDIT, 0.05) <= EPS_TARGET], default=-1)
print(f"\nThe sealed AUDIT split has Q = {Q_AUDIT} face instances; at delta=0.05 it could certify L <= 1% with at most C = {C_MAX_AUDIT} misses "
      f"(C=2 would need {SIZING_DF.loc[2, 'Q_min_delta0.05']}).")
check("2.3", "the sealed AUDIT split holds >= 299 face instances (the zero-miss requirement for a 99% claim at delta=0.05; a-priori)",
      Q_AUDIT >= 299, f"Q={Q_AUDIT}; max C for L<=1% = {C_MAX_AUDIT}")

# ---- (c) footage minutes and labelling cost: ASSUMPTIONS (the human replaces them) ----
INSTANCES_PER_MINUTE_STAGED = 1.0            # ASSUMPTION: staged capture, people deliberately entering frame ~1 face instance per footage minute
ASSUMED_LABEL_USD_PER_VIDEO_MINUTE = 1.50    # ASSUMPTION: placeholder labelling price per footage minute; NOT a vendor quote
NATURAL_DENSITY_PER_MIN = n_inst_main / float(SPLIT_DF["simulated_minutes"].sum())   # synthetic: measured on this corpus
minute_rows = []
for C in [0, 1, 2]:
    q = int(SIZING_DF.loc[SIZING_DF["allowed_misses_C"] == C, "Q_min_delta0.05"].iloc[0])
    minute_rows.append(dict(allowed_misses_C=C, Q_min=q, staged_minutes_assumed=q / INSTANCES_PER_MINUTE_STAGED,
                            natural_minutes_synthetic_density=q / NATURAL_DENSITY_PER_MIN,
                            staged_label_usd_assumed=q / INSTANCES_PER_MINUTE_STAGED * ASSUMED_LABEL_USD_PER_VIDEO_MINUTE,
                            natural_label_usd_assumed=q / NATURAL_DENSITY_PER_MIN * ASSUMED_LABEL_USD_PER_VIDEO_MINUTE))
MINUTES_DF = pd.DataFrame(minute_rows).round(1)
print(f"\nfootage and labelling cost (ASSUMED: {INSTANCES_PER_MINUTE_STAGED} instance/min staged, ${ASSUMED_LABEL_USD_PER_VIDEO_MINUTE}/min labelling; "
      f"SYNTHETIC natural density {NATURAL_DENSITY_PER_MIN:.3f} instances/min):")
print(MINUTES_DF.to_string(index=False))

# ---- (d) DERIVED here: which face kinds the staged set should over-sample (per-kind miss measured on TUNE, one reference config) ----
REF_KEY = (0.7, 1, 30, 20, 0.3)             # reference config used only to RANK kinds by difficulty (at thr 0.6 only one kind misses anything, so the ranking is uninformative)
_ref = evaluate(CORP["TUNE"], run_pipeline(DETS["TUNE"], CORP["TUNE"], pc_of(REF_KEY)))
KIND_TUNE = _ref["per_kind"].set_index("kind").reindex(FACE_KINDS)
KIND_TUNE["miss_rate"] = 1 - KIND_TUNE["instance_recall"]
_w_nat = KIND_TUNE["n_instances"] / KIND_TUNE["n_instances"].sum()
_floor, _n_min_kind = 0.02, 30               # a-priori: miss-rate floor (no kind is treated as perfect) and >= 30 instances per kind
_raw = _w_nat * np.sqrt((KIND_TUNE["miss_rate"] + _floor) * (1 - KIND_TUNE["miss_rate"]))   # Neyman-style: n_k ~ w_k * sd_k
Q_STAGED_TOTAL = int(SIZING_DF.loc[SIZING_DF["allowed_misses_C"] == 2, "Q_min_delta0.05"].iloc[0])           # size for C=2
_alloc = np.maximum(_n_min_kind, _raw / _raw.sum() * Q_STAGED_TOTAL)
_alloc = _alloc / _alloc.sum() * Q_STAGED_TOTAL
COMP_DF = pd.DataFrame(dict(kind=FACE_KINDS, tune_instances=KIND_TUNE["n_instances"].to_numpy(), tune_miss_rate=KIND_TUNE["miss_rate"].round(3).to_numpy(),
                            natural_share=_w_nat.round(3).to_numpy(), staged_share=(_alloc / _alloc.sum()).round(3).to_numpy(),
                            staged_instances=np.round(_alloc).astype(int).to_numpy()))
print(f"\nrecommended staged composition for Q = {Q_STAGED_TOTAL} (derived here; per-kind TUNE miss rates at reference config {REF_KEY}):")
print(COMP_DF.to_string(index=False))
_worst = COMP_DF.sort_values("tune_miss_rate").iloc[-1]
check("2.4", "the staged composition over-samples the kind with the highest TUNE miss rate relative to its natural share, and every kind gets >= 30 instances "
      "(a-priori design rules)",
      _worst["staged_share"] > _worst["natural_share"] and (COMP_DF["staged_instances"] >= 30).all() and abs(COMP_DF["staged_instances"].sum() - Q_STAGED_TOTAL) <= len(FACE_KINDS),
      f"worst kind {_worst['kind']}: natural {_worst['natural_share']:.3f} -> staged {_worst['staged_share']:.3f}; total {COMP_DF['staged_instances'].sum()}")
