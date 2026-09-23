# ---- (b) Excluded-pool missed-mass certificate (Thm 7 binomial / Thm 8 hypergeometric) [P3 Sec. 4.1-4.2, facered-q7] ----
# ITEMS (derived here): fixed-length 45-second video segments of the audit episodes. g(item) = 1 if the pipeline drew >= 1 blur box in the segment, else 0
# (the EXCLUDED pool: segments the pipeline left completely untouched). Y(item) = 1 if the segment contains >= 1 visible face-to-redact frame. A positive
# excluded segment is therefore a segment with an unredacted face frame. Segments of the same face track / episode are DEPENDENT (see the coverage study below).
SEG = SEG_SECONDS * 30
def segment_items(corpus: dict, bb: pd.DataFrame) -> pd.DataFrame:
    """One row per (episode, segment): n_seg items; `has_box` (g=1), `positive` (segment holds >= 1 visible face-to-redact GT frame)."""
    ep = corpus["episodes"]
    n_seg = np.ceil(ep["duration_frames"].to_numpy() / SEG).astype(int)
    items = pd.DataFrame(dict(episode_id=np.repeat(ep["episode_id"].to_numpy(), n_seg), seg=np.concatenate([np.arange(k) for k in n_seg])))
    cache = corpus["_gt_all"][1]
    face = corpus["tracks"].loc[corpus["tracks"]["is_face_to_redact"], "track_id"]
    gt = cache[cache["track_id"].isin(face)]
    pos = gt.assign(seg=gt["frame"] // SEG)[["episode_id", "seg"]].drop_duplicates().assign(positive=True)
    box = bb.assign(seg=bb["frame"] // SEG)[["episode_id", "seg"]].drop_duplicates().assign(has_box=True) if len(bb) else pd.DataFrame(columns=["episode_id", "seg", "has_box"])
    items = items.merge(pos, on=["episode_id", "seg"], how="left").merge(box, on=["episode_id", "seg"], how="left")
    items["positive"] = items["positive"].fillna(False).astype(bool); items["has_box"] = items["has_box"].fillna(False).astype(bool)
    return items

def hypergeom_upper(K0, N0, n0, alpha):
    """M_U(K0, alpha; N0, n0) = max{m in [0,N0] : P(Hypergeom(N0, m, n0) <= K0) >= alpha} [P3 Thm 8, facered-q7]; binary search (cdf is non-increasing in m)."""
    lo, hi = K0, N0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if stats.hypergeom.cdf(K0, N0, mid, n0) >= alpha:
            lo = mid
        else:
            hi = mid - 1
    return lo

ITEMS = segment_items(CORP["AUDIT"], AUD["T2"]["bb"])
N_ALL, N_EXC = len(ITEMS), int((~ITEMS["has_box"]).sum())
EXC = ITEMS[~ITEMS["has_box"]].reset_index(drop=True)
M0_TRUE, PI_HAT = int(EXC["positive"].sum()), float(ITEMS["positive"].mean())
print(f"AUDIT segments (45 s): N = {N_ALL} items; excluded pool (no blur box drawn) N0 = {N_EXC} ({N_EXC / N_ALL:.1%}); positives in excluded pool M0 = {M0_TRUE} "
      f"(eta = {M0_TRUE / max(N_EXC, 1):.4f}, r = M0/N = {M0_TRUE / N_ALL:.4f}); face-holding segments overall: {PI_HAT:.3f} of all segments")

delta0 = delta1 = DELTA_AUDIT / 2                 # delta split between the missed-mass bound and the prevalence lower bound (union bound)
rng_b = np.random.default_rng(SEED + 21)
draw = rng_b.choice(N_EXC, size=N0_SAMPLE, replace=False)                          # n0 segments uniformly WITHOUT replacement from the excluded pool
K0 = int(EXC["positive"].to_numpy()[draw].sum())
M_U = hypergeom_upper(K0, N_EXC, N0_SAMPLE, delta0)
U_bin = float(cp_upper(K0, N0_SAMPLE, delta0))
r_hyper, r_binom = M_U / N_ALL, (N_EXC / N_ALL) * U_bin
print(f"\naudit sample: n0 = {N0_SAMPLE} excluded segments labelled, K0 = {K0} positives")
print(f"  Thm 8 (hypergeometric, finite pool): M0 <= M_U = {M_U}  =>  r(g) <= {r_hyper:.4f}   [true r = {M0_TRUE / N_ALL:.4f}]")
print(f"  Thm 7 (binomial): eta <= U_300({K0}, {delta0}) = {U_bin:.4f}  =>  r(g) <= p0 * U = {r_binom:.4f}")
# Remark 9: absolute missed mass is NOT recall. Recall >= 1 - u/a needs a certified lower bound a on the prevalence pi; here from n = 300 uniformly drawn segments.
draw_pi = rng_b.choice(N_ALL, size=N_PI_SAMPLE, replace=False)
K_pi = int(ITEMS["positive"].to_numpy()[draw_pi].sum())
a_pi = float(cp_lower(K_pi, N_PI_SAMPLE, delta1))
recall_seg_lb = max(0.0, 1 - r_hyper / a_pi)
print(f"  Remark 9: prevalence sample K = {K_pi}/{N_PI_SAMPLE}, certified lower bound a = L(K, {delta1}) = {a_pi:.3f}; implied SEGMENT-level recall >= 1 - r_bound/a = {recall_seg_lb:.3f} "
      f"(joint confidence {1 - delta0 - delta1:.3f}); true segment recall = {1 - M0_TRUE / max(int(ITEMS['positive'].sum()), 1):.3f}")
print(f"  So an absolute missed-mass bound of {r_hyper:.3f} looks small, but with {PI_HAT:.0%} of segments holding a face it certifies only ~{recall_seg_lb:.0%} segment recall; "
      f"on real footage where faces are rarer (P3 Remark 9: pi = 0.01, r <= 0.005 -> recall >= 0.5) it would certify far less. It is NOT the client's instance-recall claim.")

print(f"\nWhy is K0 = 0? In this pipeline a face track almost always draws SOME box in every 45-s segment it lives in (bridging + padding around any detection);\n"
      f"its misses are frames left uncovered INSIDE segments that do contain boxes (dropouts, box too small). Those live in the INCLUDED pool, where P3 Prop. 3\n"
      f"says labels give no missed-mass bound, and where only the instance-level Prop. 26 audit of (a) speaks. The excluded-pool audit certifies a different, easier event.")

# --- Monte Carlo coverage on the SIMULATOR (TRUTH population; the sealed AUDIT labels are not used again) ---
# For the excluded pool to contain positives at all, the coverage study uses a deliberately WEAK config (thr 0.95): a face whose whole track stays
# under the threshold draws no box, so its segments are excluded AND positive.
WEAK_KEY = (0.95, 1, 30, 20, 0.3)
t0 = time.time()
_tr_items = []
for _n, _c in TRUTH_CH.items():
    _it = segment_items(_c, run_pipeline(TRUTH_CH_DET[_n], _c, pc_of(WEAK_KEY)))
    _it["ep_key"] = _n + "_" + _it["episode_id"].astype(str)
    _tr_items.append(_it)
TR_ITEMS = pd.concat(_tr_items, ignore_index=True)
exc_mask = ~TR_ITEMS["has_box"].to_numpy(); pos_all = TR_ITEMS["positive"].to_numpy()
N_TR, N0_TR = len(TR_ITEMS), int(exc_mask.sum()); M0_TR = int((pos_all & exc_mask).sum()); eta_pop = M0_TR / N0_TR
print(f"\nTRUTH ({N_TRUTH} episodes, {N_TR} segments) with the weak config {describe_key(WEAK_KEY)}: excluded pool {N0_TR} ({N0_TR / N_TR:.1%}), positives M0 = {M0_TR}, "
      f"eta_pop = {eta_pop:.4f}, r = {M0_TR / N_TR:.4f}  [{time.time() - t0:.1f}s]")
R_MC = 3000
pos_exc = pos_all[exc_mask]
# (i) design-based (Thm 8): repeat the n0-draw on the FIXED finite excluded pool; valid by the sampling design whatever the dependence between segments
K_mc = rng_b.hypergeometric(M0_TR, N0_TR - M0_TR, N0_SAMPLE, size=R_MC)
_MU = {int(k): hypergeom_upper(int(k), N0_TR, N0_SAMPLE, delta0) for k in np.unique(K_mc)}
cov_thm8 = float(np.mean([M0_TR <= _MU[int(k)] for k in K_mc]))
tol_mc = 3 * np.sqrt(delta0 * (1 - delta0) / R_MC)
check("5.3", f"Thm 8 (hypergeometric) coverage on the finite TRUTH excluded pool over {R_MC} repeated draws of n0={N0_SAMPLE}: P(M0 <= M_U) >= 1 - delta0 - 3 SE = "
      f"{1 - delta0 - tol_mc:.4f} (a-priori)", cov_thm8 >= 1 - delta0 - tol_mc, f"coverage {cov_thm8:.4f} (true M0 = {M0_TR}, typical M_U = {int(np.median(list(_MU.values())))})")
# (ii) population-level Thm 7 with DEPENDENT segments: each replicate audit sees only E_SUB = 600 episodes (random subset), n0 = N0_POP = 1000 excluded segments from them
uniq_ep, ep_inv = np.unique(TR_ITEMS["ep_key"].to_numpy(), return_inverse=True)
by_ep_exc = [np.where((ep_inv == e) & exc_mask)[0] for e in range(len(uniq_ep))]
R_POP, E_SUB, N0_POP = 600, 600, 1000
hits = []
for _ in range(R_POP):
    sub = rng_b.choice(len(uniq_ep), size=E_SUB, replace=False)
    pool = np.concatenate([by_ep_exc[e] for e in sub])
    take = rng_b.choice(pool, size=min(N0_POP, len(pool)), replace=False)
    hits.append(eta_pop <= float(cp_upper(int(pos_all[take].sum()), len(take), delta0)))
cov_thm7 = float(np.mean(hits)); tol_p = 3 * np.sqrt(delta0 * (1 - delta0) / R_POP)
print(f"Thm 7 (binomial, i.i.d. items assumed) on DEPENDENT segments: audits of {N0_POP} excluded segments from {E_SUB}-episode subsets: "
      f"coverage of eta_pop <= U = {cov_thm7:.4f} (nominal {1 - delta0:.3f})")
check("5.4", f"Thm 7 binomial coverage of the population excluded-pool rate when the audited segments come from only {E_SUB} episodes (dependent segments; weak config), over {R_POP} "
      f"replicate audits: >= 1 - delta0 - 3 SE = {1 - delta0 - tol_p:.3f} (a-priori; an honest FAIL would mean clustering breaks the i.i.d. assumption)",
      cov_thm7 >= 1 - delta0 - tol_p, f"coverage {cov_thm7:.4f}")
