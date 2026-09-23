# ---- 7. Re-verification harness: a compact function the client (or you) re-runs whenever the model, threshold or footage changes ----
def reverify(config: dict, labelled_corpus: dict, detections: pd.DataFrame, delta: float, targets: dict, n_boot: int = 1000, seed: int = 0) -> dict:
    """Re-verify ONE pipeline configuration against a FRESH labelled audit set.

    Inputs: `config` (NB2 pipeline-config dict), a labelled corpus (episodes + latent/GT truth), the raw detector output for those episodes, the error level `delta`,
    and `targets` = dict(miss=alpha_miss, fb=alpha_fb, client_recall=0.99, client_precision=0.95).

    What it CERTIFIES (only if `config` and `targets` were fixed BEFORE these labels were examined and the episodes are a fresh, unopened sample):
      * Prop. 26 [P3]: with prob >= 1 - delta, the instance miss rate L <= U_Q(C, delta) (Clopper-Pearson over the Q audited face INSTANCES) -> a recall lower bound;
      * LTT p-values [P1] for both risks (binomial for the binary miss loss, Hoeffding-Bentkus for the episode-averaged false-blur fraction) and their max (Prop. 6);
      * a verdict: PASS / FAIL / INCONCLUSIVE for the LTT targets and, separately, for the client's 99% recall / 95% pooled-precision claims.
    What it does NOT certify: any other configuration; a config or threshold changed after these labels were seen (that needs a NEW audit set); footage unlike this generator
    (distribution shift); temporal dependence beyond 'instances are exchangeable'; noisy labels; the cluster-bootstrap precision interval is a bootstrap, not a paper guarantee.
    Verdict rule (fixed here): PASS = both p-values <= delta; FAIL = the data reject the target (Clopper-Pearson lower bound of the miss rate > alpha_miss, or the Hoeffding lower
    bound of the false-blur risk > alpha_fb); INCONCLUSIVE otherwise (reason given, e.g. Q too small for the target even with zero misses)."""
    L = episode_losses(labelled_corpus, detections, config)
    Q, C = int(L["n_inst"].sum()), int(L["n_miss"].sum()); n_ep = len(L["n_inst"])
    fdp = np.where(L["n_box"] > 0, L["n_false"] / np.maximum(L["n_box"], 1), 0.0); R_fb = float(fdp.mean())
    U, Lo = float(cp_upper(C, Q, delta)), float(cp_lower(C, Q, delta))
    p_m, p_f = float(p_binom(C, Q, targets["miss"])), float(p_hb(R_fb, n_ep, targets["fb"]))
    hoeff_lb = R_fb - float(np.sqrt(np.log(1 / delta) / (2 * n_ep)))
    idx = np.random.default_rng(seed).integers(0, n_ep, size=(n_boot, n_ep))                      # episode-cluster bootstrap
    prec_b = 1 - L["n_false"][idx].sum(1) / np.maximum(L["n_box"][idx].sum(1), 1)
    prec, prec_ci = float(1 - L["n_false"].sum() / max(L["n_box"].sum(), 1)), np.percentile(prec_b, [2.5, 97.5])
    q_zero = int(np.ceil(np.log(1 / delta) / np.log(1 / (1 - targets["miss"]))))                   # instances needed even with ZERO misses to certify alpha_miss
    if max(p_m, p_f) <= delta:
        v_ltt, why = "PASS", "both p-values <= delta"
    elif Lo > targets["miss"] or hoeff_lb > targets["fb"]:
        v_ltt, why = "FAIL", "data reject the target: " + ("miss-rate lower bound %.4f > alpha_miss" % Lo if Lo > targets["miss"] else "false-blur lower bound %.3f > alpha_fb" % hoeff_lb)
    else:
        v_ltt, why = "INCONCLUSIVE", ("Q=%d < %d needed even with zero misses" % (Q, q_zero)) if Q < q_zero else "p-values above delta but no evidence of violation (borderline / more data needed)"
    q_client = int(np.ceil(np.log(1 / delta) / np.log(1 / (1 - (1 - targets["client_recall"])))))
    r_ok, r_bad = U <= 1 - targets["client_recall"], Lo > 1 - targets["client_recall"]
    p_ok, p_bad = prec_ci[0] >= targets["client_precision"], prec_ci[1] < targets["client_precision"]
    v_client = "FAIL" if (r_bad or p_bad) else "PASS" if (r_ok and p_ok) else "INCONCLUSIVE"
    return dict(config=dict(config), Q=Q, C=C, n_episodes=n_ep, miss_rate=C / max(Q, 1), prop26_U=U, recall_lower_bound=1 - U, miss_rate_lower_bound=Lo,
                Q_needed_zero_miss_for_alpha_miss=q_zero, Q_needed_zero_miss_for_client_recall=q_client, episode_avg_false_blur=R_fb,
                pooled_precision=prec, pooled_precision_boot95=[float(prec_ci[0]), float(prec_ci[1])], p_miss=p_m, p_fb=p_f, p_max=max(p_m, p_f),
                verdict_ltt_targets=v_ltt, verdict_reason=why, verdict_client_claims=v_client, delta=delta, targets=dict(targets))

HARNESS_TARGETS = dict(miss=ALPHA_T2["miss"], fb=ALPHA_T2["fb"], client_recall=0.99, client_precision=0.95)
def fresh_set(seed, n_episodes, det=None):
    c = generate_corpus(SimConfig(n_episodes=n_episodes, seed=seed))
    return c, simulate_detector(c, det or {}, seed=seed + 1)
CFG_T2 = pc_of(ALL_KEYS[J_T2]); CFG_T2_HI = dict(CFG_T2, score_thr=0.90); CFG_T2_LO = dict(CFG_T2, score_thr=0.60)
t0 = time.time()
H = {}
c_, d_ = fresh_set(SEED + 50, 500);                       H["(i) unchanged config, fresh audit set"] = reverify(CFG_T2, c_, d_, DELTA_AUDIT, HARNESS_TARGETS)
c_, d_ = fresh_set(SEED + 51, 500, dict(base=0.0));       H["(ii) model change: detector base logit -1.0, fresh audit set"] = reverify(CFG_T2, c_, d_, DELTA_AUDIT, HARNESS_TARGETS)
c_, d_ = fresh_set(SEED + 52, 500);                       H["(iii-a) threshold change 0.70 -> 0.90, fresh audit set"] = reverify(CFG_T2_HI, c_, d_, DELTA_AUDIT, HARNESS_TARGETS)
c_, d_ = fresh_set(SEED + 53, 500);                       H["(iii-b) threshold change 0.70 -> 0.60, fresh audit set"] = reverify(CFG_T2_LO, c_, d_, DELTA_AUDIT, HARNESS_TARGETS)
c_, d_ = fresh_set(SEED + 54, 40);                        H["(iv) unchanged config, SMALL fresh audit set (40 episodes)"] = reverify(CFG_T2, c_, d_, DELTA_AUDIT, HARNESS_TARGETS)
print(f"5 harness demonstrations in {time.time() - t0:.1f}s")
HDF = pd.DataFrame([dict(case=k, Q=v["Q"], C=v["C"], miss_rate=round(v["miss_rate"], 4), recall_LB=round(v["recall_lower_bound"], 4), R_fb_episode=round(v["episode_avg_false_blur"], 3),
                         pooled_precision=round(v["pooled_precision"], 3), p_miss=round(v["p_miss"], 4), p_fb=round(v["p_fb"], 4), verdict_LTT_targets=v["verdict_ltt_targets"],
                         verdict_client_99_95=v["verdict_client_claims"]) for k, v in H.items()])
pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 62)
print(f"targets: LTT T2 (miss <= {HARNESS_TARGETS['miss']}, false-blur <= {HARNESS_TARGETS['fb']}) and the client's (recall >= 0.99, pooled precision >= 0.95); delta = {DELTA_AUDIT}"); print(HDF.to_string(index=False))
for k, v in H.items():
    print(f"  {k}: LTT verdict {v['verdict_ltt_targets']} ({v['verdict_reason']}); client claims {v['verdict_client_claims']}; Q needed for a 99% claim with 0 misses = {v['Q_needed_zero_miss_for_client_recall']}")

# regression test of the harness against section 5 (same sealed audit, same config: must reproduce Prop. 26 exactly)
_rv = reverify(CFG_T2, CORP["AUDIT"], DETS["AUDIT"], DELTA_AUDIT, HARNESS_TARGETS)
_cases = list(H.values())
check("7.1", "harness regression: on the sealed AUDIT with the T2 config, reverify() reproduces section 5's Q, C and Prop. 26 bound exactly",
      (_rv["Q"], _rv["C"]) == (AUD["T2"]["Q"], AUD["T2"]["C"]) and abs(_rv["prop26_U"] - AUD["T2"]["U"]) < 1e-12, f"Q={_rv['Q']}, C={_rv['C']}, U={_rv['prop26_U']:.4f}")
check("7.2", "(i) the unchanged config on a fresh audit set is NOT flagged: LTT-target verdict PASS (a-priori claim)", _cases[0]["verdict_ltt_targets"] == "PASS",
      f"verdict {_cases[0]['verdict_ltt_targets']}: {_cases[0]['verdict_reason']} (miss {_cases[0]['C']}/{_cases[0]['Q']}, p_max {_cases[0]['p_max']:.4f})")
check("7.3", "(ii) the model change is CAUGHT: LTT-target verdict FAIL (a-priori claim)", _cases[1]["verdict_ltt_targets"] == "FAIL",
      f"verdict {_cases[1]['verdict_ltt_targets']}: {_cases[1]['verdict_reason']} (miss {_cases[1]['C']}/{_cases[1]['Q']})")
check("7.4", "(iii-a) the aggressive threshold change 0.70 -> 0.90 is CAUGHT: LTT-target verdict FAIL (a-priori claim)", _cases[2]["verdict_ltt_targets"] == "FAIL",
      f"verdict {_cases[2]['verdict_ltt_targets']}: {_cases[2]['verdict_reason']} (miss {_cases[2]['C']}/{_cases[2]['Q']})")
check("7.5", "(iv) a small audit set (Q far below the 299 zero-miss requirement) never gets PASS for the client's 99% recall claim: the verdict is INCONCLUSIVE, or FAIL when the few audited misses already reject 99% (a-priori)",
      _cases[4]["verdict_client_claims"] != "PASS" and _cases[4]["Q"] < 299, f"Q={_cases[4]['Q']}, client verdict {_cases[4]['verdict_client_claims']}")
_rv2 = reverify(CFG_T2, CORP["AUDIT"], DETS["AUDIT"], DELTA_AUDIT, HARNESS_TARGETS)
report = dict(protocol="Any design / threshold / model change requires a FRESH sealed audit set; certificates hold only for a config fixed before its labels were opened [P3 Assumption 1, Sec. 2.4].",
              targets=HARNESS_TARGETS, delta=DELTA_AUDIT, cases=H, regression_sealed_audit=_rv)
(NB3_DIR / "reverify_report.json").write_text(json.dumps(report, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
_back = json.loads((NB3_DIR / "reverify_report.json").read_text())
check("7.6", "the JSON report is written to data/nb3/, round-trips, and reverify() is deterministic (two runs on the same inputs give identical dicts)",
      _back["regression_sealed_audit"]["C"] == _rv["C"] and _rv == _rv2, f"{(NB3_DIR / 'reverify_report.json').stat().st_size} bytes")
print("\nProtocol: any design / threshold / model change requires a fresh sealed audit set; certificates are valid only for a config fixed before its labels were opened. "
      "Re-using opened labels 'voids the guarantee' [P3 Sec. 2.4].")
