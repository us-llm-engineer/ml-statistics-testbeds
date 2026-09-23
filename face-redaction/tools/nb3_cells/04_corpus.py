# ---- the larger corpus: SAME generator and distribution as NB2, more episodes ----
N_MAIN, N_TUNE, N_CAL = 2300, 650, 950      # MAIN = TUNE + CAL + AUDIT (the audit gets the remaining 700)
N_TRUTH = 2400                              # separate large held-out population, used ONLY to approximate "true" risks
MAIN_SEED, TRUTH_SEED, DET_SEED, TRUTH_DET_SEED = SEED + 1, SEED + 2, SEED + 3, SEED + 4

def subset_corpus(corpus: dict, episode_ids) -> dict:
    """Corpus restricted to `episode_ids`: episodes, tracks and latent truth are filtered. The returned dict
    has NO `_gt_all` cache and a NEW latent dict, so `evaluate` builds a fresh ground-truth cache keyed to
    this subset (NB2's cache is keyed by id(latent)); nothing is shared with the parent's cache."""
    ids = {int(e) for e in episode_ids}
    eps = corpus["episodes"][corpus["episodes"]["episode_id"].isin(ids)].reset_index(drop=True)
    trk = corpus["tracks"][corpus["tracks"]["episode_id"].isin(ids)].reset_index(drop=True)
    lat = {int(t): corpus["latent"][int(t)] for t in trk["track_id"]}
    return dict(episodes=eps, tracks=trk, latent=lat, cfg=corpus["cfg"])

def dets_for(detections: pd.DataFrame, corpus: dict) -> pd.DataFrame:
    return detections[detections["episode_id"].isin(set(corpus["episodes"]["episode_id"]))].reset_index(drop=True)

t0 = time.time(); MAIN = generate_corpus(SimConfig(n_episodes=N_MAIN, seed=MAIN_SEED)); t_gen = time.time() - t0
t0 = time.time(); MAIN_DET = simulate_detector(MAIN, {}, seed=DET_SEED); t_det = time.time() - t0
t0 = time.time(); TRUTH = generate_corpus(SimConfig(n_episodes=N_TRUTH, seed=TRUTH_SEED)); t_gen_t = time.time() - t0
t0 = time.time(); TRUTH_DET = simulate_detector(TRUTH, {}, seed=TRUTH_DET_SEED); t_det_t = time.time() - t0
print(f"MAIN : generate {t_gen:.2f}s, detect {t_det:.2f}s ({len(MAIN_DET):,} raw detections, {N_MAIN} episodes)")
print(f"TRUTH: generate {t_gen_t:.2f}s, detect {t_det_t:.2f}s ({len(TRUTH_DET):,} raw detections, {N_TRUTH} episodes)")

# three disjoint EPISODE-level splits (random, fixed seed)
_perm = np.random.default_rng(SEED + 5).permutation(MAIN["episodes"]["episode_id"].to_numpy())
SPLIT_IDS = dict(TUNE=np.sort(_perm[:N_TUNE]), CAL=np.sort(_perm[N_TUNE:N_TUNE + N_CAL]), AUDIT=np.sort(_perm[N_TUNE + N_CAL:]))
CORP = {k: subset_corpus(MAIN, v) for k, v in SPLIT_IDS.items()}
DETS = {k: dets_for(MAIN_DET, CORP[k]) for k in CORP}
# TRUTH is processed in 4 chunks of equal size (parallel tasks; per-episode losses are independent across episodes)
_tperm = TRUTH["episodes"]["episode_id"].to_numpy()
TRUTH_CH = {f"TRUTH{k}": subset_corpus(TRUTH, _tperm[k * (N_TRUTH // 4):(k + 1) * (N_TRUTH // 4)]) for k in range(4)}
TRUTH_CH_DET = {k: dets_for(TRUTH_DET, v) for k, v in TRUTH_CH.items()}

def split_table(corpora: dict) -> pd.DataFrame:
    rows = []
    for name, c in corpora.items():
        tr, ep = c["tracks"], c["episodes"]
        rows.append(dict(split=name, episodes=len(ep), zero_face_episodes=int((ep["n_face_tracks"] == 0).sum()),
                         face_instances=int(tr["is_face_to_redact"].sum()),
                         simulated_minutes=round(float(ep["duration_frames"].sum() / ep["fps"].iloc[0] / 60), 0)))
    return pd.DataFrame(rows)
SPLIT_DF = split_table({k: CORP[k] for k in ("TUNE", "CAL", "AUDIT")})
print(); print(SPLIT_DF.to_string(index=False))
n_inst_main = int(SPLIT_DF["face_instances"].sum())

# --- CHECK 1.x ---
check("1.1", "face-to-redact instances across TUNE+CAL+AUDIT >= 1,500 (a-priori; NB2's 222 cannot support a 99% certificate)",
      n_inst_main >= 1500, f"{n_inst_main} instances in {N_MAIN} episodes (NB2: 222 in 300)")
_all = np.concatenate(list(SPLIT_IDS.values()))
check("1.2", "the three splits are pairwise disjoint at the EPISODE level and together cover all MAIN episodes",
      len(set(_all.tolist())) == len(_all) == N_MAIN and all(
          not (set(SPLIT_IDS[a].tolist()) & set(SPLIT_IDS[b].tolist())) for a in SPLIT_IDS for b in SPLIT_IDS if a < b),
      f"sizes {[len(v) for v in SPLIT_IDS.values()]}, union {len(set(_all.tolist()))}")
_zero = float((MAIN["episodes"]["n_face_tracks"] == 0).mean())
_tr_main = MAIN["tracks"]
_kind_share_main = _tr_main.loc[_tr_main["is_face_to_redact"], "kind"].value_counts(normalize=True).reindex(FACE_KINDS).fillna(0)
_w = pd.Series({k: KIND_PARAMS[k][-1] for k in FACE_KINDS}); _w = _w / _w.sum()      # NB2's design weights among redacted kinds
_se = np.sqrt(_w * (1 - _w) / n_inst_main)
_z = ((_kind_share_main - _w) / _se).abs()
check("1.3", "the larger corpus has NB2's distribution: zero-face-episode fraction in [0.45,0.60] and every face-kind share within 3 binomial SE of "
      "NB2's design weights (a-priori; a 3-SE band on 7 kinds fails by chance ~2%)",
      0.45 <= _zero <= 0.60 and float(_z.max()) <= 3.0, f"zero-face {_zero:.3f}; max |z| over kinds {float(_z.max()):.2f}")
# Informational only (not a check): NB2's own 222-instance sample vs the same design weights. An earlier draft of this check
# compared the new corpus to NB2's realised shares within +-0.05; that failed (0.065) because NB2's n=222 sample has SE ~0.03 per kind.
_tr_nb2 = NB2_CORPUS["tracks"]
_sh_nb2 = _tr_nb2.loc[_tr_nb2["is_face_to_redact"], "kind"].value_counts(normalize=True).reindex(FACE_KINDS).fillna(0)
print(f"(info) NB2's own 222-instance sample: max |z| vs design weights = {float((((_sh_nb2 - _w) / np.sqrt(_w * (1 - _w) / 222)).abs()).max()):.2f}; "
      f"max |share diff| new-vs-NB2 realised = {float((_kind_share_main - _sh_nb2).abs().max()):.3f}")

# CAL2: a SECOND, fresh calibration split from the same generator (new seeds), held back for section 3(c). Only its label-free detector
# output exists until then; its labels are first touched (evaluate) in section 3(c). Reason: see 3(b)-(c) -- a first CAL split can be consumed.
N_CAL2, CAL2_SEED, CAL2_DET_SEED = 950, SEED + 6, SEED + 7
_c2 = generate_corpus(SimConfig(n_episodes=N_CAL2, seed=CAL2_SEED))
CORP["CAL2"], DETS["CAL2"] = _c2, simulate_detector(_c2, {}, seed=CAL2_DET_SEED)
print(f"CAL2 (fresh, held back): {N_CAL2} episodes, {int(_c2['tracks']['is_face_to_redact'].sum())} face instances")
