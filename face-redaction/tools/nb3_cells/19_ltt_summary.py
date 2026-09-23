# ---- main-run T1/T2 outcomes, the chosen T2 config and the cost of certification (numbers only; charts come next) ----
_lam, _det = LTT_MAIN["T2"]["lam_hat"], LTT_MAIN["T2"]
J_T2_MAIN = pick_cheapest(_lam, _det["p_test"])
cert_idx = np.where(_lam)[0]
cert_cost = COST_VEC[cert_idx]
most_conservative = int(cert_idx[np.argmin(_det["p_test"]["miss"][cert_idx])]) if len(cert_idx) else -1        # lowest CAL2 miss point estimate in Lambda-hat
print(f"T1 (client: miss <= {ALPHA_T1['miss']}, false-blur <= {ALPHA_T1['fb']}, delta {DELTA}): |Lambda-hat| = {int(LTT_MAIN['T1']['lam_hat'].sum())} -> NO CERTIFIED CONFIGURATION on CAL1")
p1 = LTT_MAIN["T1"]["p_test"]; exc1 = np.maximum(p1["miss"] / ALPHA_T1["miss"], p1["fb"] / ALPHA_T1["fb"]); jb = int(np.argmin(exc1))
print(f"   best CAL1 point estimates against T1 (NOT certified): closest config {describe_key(ALL_KEYS[jb])}: miss {p1['miss'][jb]:.4f}, FDP {p1['fb'][jb]:.3f}, pooled precision {p1['pooled_prec'][jb]:.3f}; "
      f"lowest miss anywhere {p1['miss'].min():.4f} (that config's FDP {p1['fb'][int(np.argmin(p1['miss']))]:.3f}); "
      f"lowest FDP among configs with miss <= 1%: {p1['fb'][p1['miss'] <= 0.01].min():.3f} -- {p1['fb'][p1['miss'] <= 0.01].min() / ALPHA_T1['fb']:.0f}x the client's {ALPHA_T1['fb']}")
print(f"T2 (miss <= {ALPHA_T2['miss']}, false-blur <= {ALPHA_T2['fb']}): |Lambda-hat| = {len(cert_idx)} on CAL2; chosen (cheapest, ties -> fewest false blurs) = {describe_key(ALL_KEYS[J_T2_MAIN])}")
p2 = _det["p_test"]
print(f"   chosen config CAL2 estimates: miss {p2['miss'][J_T2_MAIN]:.4f}, episode-FDP {p2['fb'][J_T2_MAIN]:.3f}, pooled precision {p2['pooled_prec'][J_T2_MAIN]:.3f}, p_max {p2['p_max'][J_T2_MAIN]:.4f}")
save = float(COST_VEC[most_conservative] - COST_VEC[J_T2_MAIN])
print(f"   cost (CPU $/video-hour = MEASURED throughput x ASSUMED price): chosen ${COST_VEC[J_T2_MAIN]:.3f} vs most conservative certified config ({describe_key(ALL_KEYS[most_conservative])}) "
      f"${COST_VEC[most_conservative]:.3f} -> saving ${save:.3f}/h. Every certified config is stride 1 ({sorted({ALL_KEYS[j][1] for j in cert_idx})}), so certification brought no cost saving here.")
st = [(j, ALL_KEYS[j]) for j in range(G) if ALL_KEYS[j][1] > 1]
print(f"   configs with an UNCORRECTED CAL2 p_max <= delta: {int((p2['p_max'] <= DELTA).sum())} of {G} -- only {len(cert_idx)} are certified: the fixed-sequence walk stops at the first failure, and p-values of "
      f"configs it never reached are not certificates (testing all {G} at level delta would inflate the error).")
_ch = RS["T2"]["chosen"]; _ch = _ch[_ch >= 0]
print(f"   over the {R_SPLITS} re-splits, when T2 certified something ({len(_ch)} runs), the cheapest certified config had stride>1 in {float(np.mean([ALL_KEYS[j][1] > 1 for j in _ch])):.0%} of runs")
print("   strided configs (the only cost lever) -- lowest CAL2 p_max first:")
print(pd.DataFrame(dict(config=[describe_key(k) for _, k in st], cal2_miss=[p2["miss"][j] for j, _ in st], cal2_fdp=[p2["fb"][j] for j, _ in st],
                        p_max_cal2=[p2["p_max"][j] for j, _ in st], cpu_usd_per_h=[COST_VEC[j] for j, _ in st]))
      .sort_values("p_max_cal2").head(6).round(4).to_string(index=False))
# quantities used by the charts
P_TUNE = LTT_MAIN["T2"]["p_tune"]; PATH = LTT_MAIN["T2"]["path"]
BETA_PATH = appD_beta_path(P_TUNE, ALPHA_T2)
print(f"\nP1 App. D's own beta-path would visit only {len(BETA_PATH)} of {G} configs on this grid (many configs have TUNE p-values pinned near 0 or 1); the rank path used here visits all {len(PATH)}.")
_ex_L0 = float(exc1.min())
SENS_EXCESS = [_ex_L0] + [max(r["best_cal_miss"] / ALPHA_T1["miss"], r["best_cal_fdp"] / ALPHA_T1["fb"]) for r in SENS_ROWS]
check("3.7", "cost saving from certification: the cheapest config in Lambda-hat is no more expensive than the most conservative one (a tautology by construction of pick_cheapest; informational, the printed saving is the finding) and the report states the "
      "measured saving explicitly (a-priori: saving >= 0)", save >= -1e-12, f"saving ${save:.3f}/h")
