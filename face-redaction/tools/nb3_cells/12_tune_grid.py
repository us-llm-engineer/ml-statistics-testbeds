# ---- TUNE-only work: evaluate ALL grid configs on TUNE (parallel), learn nothing from CAL or AUDIT ----
t0 = time.time()
register_work("TUNE", CORP["TUNE"], DETS["TUNE"])
_res = eval_many([("TUNE", k) for k in ALL_KEYS])
M_TUNE = to_matrices(_res, ["TUNE"], ALL_KEYS)
t_tune_grid = time.time() - t0
IDX_TUNE = np.arange(len(CORP["TUNE"]["episodes"]))
S_TUNE = risk_summary(M_TUNE, IDX_TUNE)
print(f"all {G} configs evaluated on TUNE ({len(IDX_TUNE)} episodes, {S_TUNE['n_inst']} instances) in {t_tune_grid:.1f}s wall on {N_WORKERS} workers")

FRONT = pd.DataFrame(dict(thr=[k[0] for k in ALL_KEYS], stride=[k[1] for k in ALL_KEYS], gap=[k[2] for k in ALL_KEYS], pad=[k[3] for k in ALL_KEYS],
                          margin=[k[4] for k in ALL_KEYS], tune_miss=S_TUNE["miss"], tune_fb=S_TUNE["fb"], tune_pooled_prec=S_TUNE["pooled_prec"]))
def pareto(df, xcol, ycol):
    d = df.sort_values([xcol, ycol]); best, keep = np.inf, []
    for i, r in d.iterrows():
        if r[ycol] < best:
            keep.append(i); best = r[ycol]
    return df.loc[keep]
FRONT_PARETO = pareto(FRONT, "tune_fb", "tune_miss")
print("TUNE Pareto frontier (false-blur risk vs miss risk; point estimates, TUNE only):")
print(FRONT_PARETO.round(3).to_string(index=False))

# T1 on TUNE (information only): best miss among configs, and the smallest fb among configs with miss <= 1%
t1_ok_tune = FRONT[(FRONT["tune_miss"] <= ALPHA_T1["miss"]) & (FRONT["tune_fb"] <= ALPHA_T1["fb"])]
print(f"\nT1 (client) on TUNE: configs with miss <= {ALPHA_T1['miss']} AND episode-FDP <= {ALPHA_T1['fb']}: {len(t1_ok_tune)} of {G}; "
      f"lowest miss anywhere {FRONT['tune_miss'].min():.4f}; lowest FDP among miss<=1%: "
      f"{FRONT.loc[FRONT['tune_miss'] <= 0.01, 'tune_fb'].min() if (FRONT['tune_miss'] <= 0.01).any() else float('nan'):.3f}")
