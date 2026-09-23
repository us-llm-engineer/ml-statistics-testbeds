# ---- LTT-CAL: open the CAL split for the first time. (In production only the configs the fixed-sequence walk VISITS would be run on CAL;
# ---- here all grid points are evaluated in parallel up front because section 3(c) re-uses them for the re-split simulation.) ----
log_event("CAL split opened")
t0 = time.time()
register_work("CAL", CORP["CAL"], DETS["CAL"])
_res = eval_many([("CAL", k) for k in ALL_KEYS])
M_CAL = to_matrices(_res, ["CAL"], ALL_KEYS)
IDX_CAL = np.arange(len(CORP["CAL"]["episodes"]))
print(f"all {G} configs evaluated on CAL ({len(IDX_CAL)} episodes, {int(M_CAL['n_inst'].sum())} instances) in {time.time() - t0:.1f}s wall")

def describe_key(k):
    return f"thr={k[0]:.2f} stride={k[1]} gap={k[2]} pad={k[3]} margin={k[4]}"

LTT_MAIN = {}
for name, alpha in [("T1", ALPHA_T1), ("T2v1", ALPHA_T2_V1)]:
    lam_hat, det = ltt_select(M_TUNE, IDX_TUNE, M_CAL, IDX_CAL, alpha, DELTA, starts=(0,))
    LTT_MAIN[name] = dict(lam_hat=lam_hat, **det)
    n_acc, n_vis = int(lam_hat.sum()), int(det["visited"].sum())
    print(f"\n=== {name}: alpha_miss={alpha['miss']}, alpha_fb={alpha['fb']}, delta={DELTA} -> |Lambda-hat| = {n_acc} of {G}; "
          f"CAL p-values evaluated for {n_vis} configs (lazy fixed-sequence walk)")
    if n_acc == 0:
        p_c = det["p_test"]
        exc = np.maximum(p_c["miss"] / alpha["miss"], p_c["fb"] / alpha["fb"])          # normalised excess over the targets (smaller = closer)
        top = np.argsort(exc)[:3]
        print(f"    NO CERTIFIED CONFIGURATION (empty Lambda-hat: abstain) [P1 Sec. 1.1]. Three CAL point estimates closest to the targets (NOT certified):")
        print(pd.DataFrame(dict(config=[describe_key(ALL_KEYS[i]) for i in top], cal_miss=p_c["miss"][top], cal_fdp=p_c["fb"][top],
                                cal_pooled_precision=p_c["pooled_prec"][top])).round(4).to_string(index=False))
        ok1 = (p_c["miss"] <= alpha["miss"]) & (p_c["fb"] <= alpha["fb"])
        print(f"    configs whose CAL point estimates meet BOTH {name} targets: {int(ok1.sum())}; lowest CAL miss {p_c['miss'].min():.4f}; "
              f"lowest CAL FDP among miss<=1%: {p_c['fb'][p_c['miss'] <= 0.01].min() if (p_c['miss'] <= 0.01).any() else float('nan'):.3f}")
    else:
        rows = []
        for j in np.where(lam_hat)[0]:
            rows.append(dict(config=describe_key(ALL_KEYS[j]), cal_miss=det["p_test"]["miss"][j], cal_fb=det["p_test"]["fb"][j],
                             cal_pooled_prec=det["p_test"]["pooled_prec"][j], p_max_cal=det["p_test"]["p_max"][j]))
        print(pd.DataFrame(rows).round(4).sort_values("cal_miss").to_string(index=False))
