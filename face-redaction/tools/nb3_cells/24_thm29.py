# ---- (d) Thm 29: stress-test cluster certificates [P3 Sec. 6.1, Defs 27-28, Thm 29, facered-q9] ----
# Stress-test generator A (DERIVED HERE, declared and written to disk before the audit): each audited face instance is re-scored under 3 harder detector calibrations
# (base logit lowered by 0.5 / 1.0 / 1.5: poor light, motion blur, mirror-like), so a cluster A(X_i) = {original, variant 1, 2, 3} of 4 variants of the SAME face.
# rho_max = P(some variant escapes) ; rho_avg = expected fraction of variants that escape. G = {T2 config, best-effort config}, |G| = 2 -> level delta / (2|G|).
t0 = time.time()
VARIANT_MISS = {}
for name in ("T2", "BEST"):
    cols = [AUD[name]["per_instance"]["instance_missed"].to_numpy()]
    for vi, b in enumerate(STRESS_BASES):
        dv = simulate_detector(CORP["AUDIT"], dict(base=b), seed=SEED + 40 + vi)
        res_v = evaluate(CORP["AUDIT"], run_pipeline(dv, CORP["AUDIT"], pc_of(AUD[name]["key"])))
        cols.append(res_v["per_instance"].sort_values("track_id")["instance_missed"].to_numpy())
    VARIANT_MISS[name] = np.stack(cols, axis=1)                                   # [Q, 4]
print(f"stress variants evaluated in {time.time() - t0:.1f}s")
rng29 = np.random.default_rng(SEED + 23)
lvl29 = DELTA_AUDIT / (2 * 2)
rows29 = []
for name, W in VARIANT_MISS.items():
    Q = W.shape[0]
    K_max = int(W.any(axis=1).sum())
    Z = rng29.integers(0, W.shape[1], size=Q)                                      # Z_i ~ Unif(A(X_i))
    K_rand = int(W[np.arange(Q), Z].sum())
    rows29.append(dict(config=name, Q=Q, miss_original=int(W[:, 0].sum()), miss_by_variant=W.sum(0).tolist(), K_max=K_max, K_rand=K_rand,
                       rho_max_bound=round(float(cp_upper(K_max, Q, lvl29)), 4), rho_avg_bound=round(float(cp_upper(K_rand, Q, lvl29)), 4)))
T29 = pd.DataFrame(rows29)
print(f"\nThm 29 at level delta/(2|G|) = {lvl29} (|G| = 2), simultaneous over both configs with probability >= 1 - {DELTA_AUDIT}:"); print(T29.to_string(index=False))
print("The certificate covers ONLY these declared perturbations [P3 Sec. 6.1]; it says nothing about lighting, blur or mirrors beyond the three declared detector shifts.")
check("5.7", "Thm 29 identities: K_max >= K_rand for both configs (a random variant escaping implies some variant escapes), and rho_max bound >= rho_avg bound",
      bool((T29["K_max"] >= T29["K_rand"]).all() and (T29["rho_max_bound"] >= T29["rho_avg_bound"]).all()),
      f"K_max {T29['K_max'].tolist()} vs K_rand {T29['K_rand'].tolist()}")
