# ---- 5. SEALED AUDIT. FIRST fix the configurations, write them to disk, and only then touch AUDIT labels [P3 Assumption 1, facered-q7] ----
DELTA_AUDIT = 0.05                                   # audit level (P3's planning table uses delta = 0.05)
# (1) the T2 choice from section 3: cheapest config in Lambda-hat on CAL2 (ties -> fewest false blurs on CAL2)
_lam2, _pt2 = LTT_MAIN["T2"]["lam_hat"], LTT_MAIN["T2"]["p_test"]
J_T2 = pick_cheapest(_lam2, _pt2)
# (2) a best-effort, RECALL-FIRST config for the client's 99% claim: lowest pooled TUNE+CAL+CAL2 miss point estimate (ties -> higher pooled precision). It is
#     NOT certified by LTT (T1 was empty): the audit is the only thing that could support a claim about it.
_pool_miss = risk_summary(M_POOL, np.arange(E_POOL))
J_BEST = int(np.lexsort((-_pool_miss["pooled_prec"], _pool_miss["miss"]))[0])
NESTED_THR = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]           # Thm 19 family: everything but the threshold fixed (gap 30, pad 20, margin 0.3)
NESTED_KEYS = [(t, 1, 30, 20, 0.3) for t in NESTED_THR]
EPS_SEG = 0.01                                       # the single pre-registered epsilon for Thm 19 (absolute missed mass over 45-s segments; derived here)
EPS_INST = ALPHA_T2["miss"]                          # pre-registered epsilon for the INSTANCE-level twin of Thm 19 (derived here) = T2's alpha_miss
SEG_SECONDS, N0_SAMPLE, N_PI_SAMPLE = 45, 300, 300
STRESS_BASES = [0.5, 0.0, -0.5]                      # stress-test generator A (declared): detector base logit lowered by 0.5/1.0/1.5 (poor light, motion blur, mirror-like)
AUDIT_REG = dict(delta_audit=DELTA_AUDIT, T2_config=dict(key=list(ALL_KEYS[J_T2])), best_effort_config=dict(key=list(ALL_KEYS[J_BEST])),
                 nested_family=[list(k) for k in NESTED_KEYS], eps_segment_missed_mass=EPS_SEG, eps_instance_miss_rate=EPS_INST, segment_seconds=SEG_SECONDS,
                 n0_excluded_sample=N0_SAMPLE, n_prevalence_sample=N_PI_SAMPLE, stress_generator_base_logit_shifts=[b - 1.0 for b in STRESS_BASES],
                 audit_episodes=int(len(CORP["AUDIT"]["episodes"])), written_at=time.strftime("%Y-%m-%d %H:%M:%S"))
(NB3_DIR / "audit_prereg.json").write_text(json.dumps(AUDIT_REG, indent=2))
log_event("audit configuration, family, epsilon and stress generator written to disk (AUDIT labels still unopened)")
print(json.dumps(AUDIT_REG, indent=2))
print(f"\nT2 choice : {describe_key(ALL_KEYS[J_T2])}   (CAL2: miss {_pt2['miss'][J_T2]:.4f}, episode-FDP {_pt2['fb'][J_T2]:.3f})")
print(f"best-effort: {describe_key(ALL_KEYS[J_BEST])}   (pooled TUNE+CAL+CAL2 miss {_pool_miss['miss'][J_BEST]:.4f}, pooled precision {_pool_miss['pooled_prec'][J_BEST]:.3f})")

# ---- open the AUDIT split: labels are touched for the first time below ----
log_event("AUDIT split opened")
register_work("AUDIT", CORP["AUDIT"], DETS["AUDIT"])
def audit_eval(corpus, dets, pc):
    bb = run_pipeline(dets, corpus, pc).reset_index(drop=True)
    res = evaluate(corpus, bb)
    return bb, res
AUD = {}
for name, j in (("T2", J_T2), ("BEST", J_BEST)):
    bb, res = audit_eval(CORP["AUDIT"], DETS["AUDIT"], pc_of(ALL_KEYS[j]))
    pi = res["per_instance"].sort_values("track_id").reset_index(drop=True)
    AUD[name] = dict(key=ALL_KEYS[j], bb=bb, res=res, per_instance=pi, Q=len(pi), C=int(pi["instance_missed"].sum()),
                     loss=episode_losses(CORP["AUDIT"], DETS["AUDIT"], pc_of(ALL_KEYS[j])))
# ---- (a) Prop. 26: conditional miss-rate certificate over audited face INSTANCES [P3 Prop. 26, facered-q9] ----
rows = []
for name, a in AUD.items():
    Q, C = a["Q"], a["C"]
    U = float(cp_upper(C, Q, DELTA_AUDIT)); Lo = float(cp_lower(C, Q, DELTA_AUDIT))
    rate = C / Q
    q_need = next((q for q in range(50, 20001) if cp_upper(int(round(rate * q)), q, DELTA_AUDIT) <= EPS_TARGET), None) if rate < EPS_TARGET else None
    a.update(U=U, L_low=Lo)
    rows.append(dict(config=name, key=describe_key(a["key"]), Q=Q, C_missed=C, miss_rate=round(rate, 4), U_Q_C_delta=round(U, 4),
                     certified_recall_lower_bound=round(1 - U, 4), claims_recall_99=bool(U <= EPS_TARGET),
                     Q_needed_at_this_miss_rate=(q_need if q_need else "never (rate >= 1%)" if rate >= EPS_TARGET else "n/a")))
P26_DF = pd.DataFrame(rows)
pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 55)
print("\n(a) Prop. 26 on the sealed AUDIT (delta = 0.05):"); print(P26_DF.to_string(index=False))
print(f"    Section 2 table: certifying L <= 1% needs Q >= {SIZING_DF['Q_min_delta0.05'].tolist()[:3]} audited instances for C = 0, 1, 2 misses; this audit has Q = {Q_AUDIT}.")
_t2 = AUD["T2"]
_names = [e[1] for e in EVENTS]
_i_reg = next(i for i, n_ in enumerate(_names) if n_.startswith("audit configuration")); _i_open = _names.index("AUDIT split opened")
check("5.1", "pre-registration order: the audit configs / family / epsilon / stress generator were written to disk BEFORE the AUDIT split was opened (event log)",
      _i_reg < _i_open, f"event {_i_reg} (registered) < event {_i_open} (audit opened); log: {[n_[:22] for n_ in _names]}")
check("5.2", "the audit is consistent with the LTT claim for the T2 choice: the one-sided 95% Clopper-Pearson LOWER bound on its miss rate does not exceed T2's alpha_miss "
      "(a-priori; a violation would mean the audit contradicts the certificate)", _t2["L_low"] <= ALPHA_T2["miss"],
      f"audit miss {_t2['C']}/{_t2['Q']} = {_t2['C'] / _t2['Q']:.4f}, lower bound {_t2['L_low']:.4f} vs alpha {ALPHA_T2['miss']}")
