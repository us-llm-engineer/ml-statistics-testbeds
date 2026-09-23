# ---- (e) The client's POOLED precision on the audit, with an episode-cluster bootstrap interval (a BOOTSTRAP, not a paper guarantee) ----
def cluster_bootstrap(n_box, n_false, B=2000, seed=SEED + 24):
    rng = np.random.default_rng(seed); E = len(n_box)
    idx = rng.integers(0, E, size=(B, E))
    prec = 1 - n_false[idx].sum(1) / np.maximum(n_box[idx].sum(1), 1)
    fdp = np.where(n_box[idx] > 0, n_false[idx] / np.maximum(n_box[idx], 1), 0.0).mean(1)          # the LTT risk (episode-averaged)
    return np.percentile(prec, [2.5, 97.5]), np.percentile(fdp, [2.5, 97.5])
rows_e = []
for name, a in AUD.items():
    ls = a["loss"]; nb, nf = ls["n_box"], ls["n_false"]
    (plo, phi), (flo, fhi) = cluster_bootstrap(nb, nf)
    pooled = 1 - nf.sum() / nb.sum(); fdp_pt = float(np.where(nb > 0, nf / np.maximum(nb, 1), 0.0).mean())
    a.update(pooled_prec=pooled, prec_ci=(plo, phi), fdp=fdp_pt, fdp_ci=(flo, fhi))
    rows_e.append(dict(config=name, pooled_precision=round(pooled, 3), boot95=f"[{plo:.3f}, {phi:.3f}]", episode_avg_FDP_LTT_risk=round(fdp_pt, 3),
                       FDP_boot95=f"[{flo:.3f}, {fhi:.3f}]", client_precision_target_0_95_met=bool(phi >= 0.95 and plo >= 0.95)))
PREC_DF = pd.DataFrame(rows_e)
print("Sealed-AUDIT precision (cluster = episode; percentile bootstrap, B=2000; NOT a P1/P2/P3 guarantee):"); print(PREC_DF.to_string(index=False))
print("NB: the LTT risk R_fb is the episode-AVERAGED false-blur fraction; the client's precision is POOLED over all blur box-frames. They differ, and both are reported.")
check("5.8", "bootstrap sanity: the pooled-precision point estimate lies inside its own 95% bootstrap interval for both configs (a-priori)",
      all(a["prec_ci"][0] <= a["pooled_prec"] <= a["prec_ci"][1] for a in AUD.values()), f"{[(round(float(a['pooled_prec']), 3), [round(float(x), 3) for x in a['prec_ci']]) for a in AUD.values()]}")
print("\nWhat if the audit fails (or shows a miss)?  The certified configuration is now a number, not a design freedom. Changing the pipeline in response to opened audit labels and re-certifying on the")
print("SAME labels 'violates Assumption 1, and voids the guarantee' [P3 Sec. 2.4]. Allowed: (i) a FRESH, unopened audit split, or (ii) a pre-registered error-spending plan across audit rounds.")
log_event("audit certificates computed; AUDIT labels now burned for design purposes")
