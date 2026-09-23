# ---- CHECK: evaluating a SUBSET corpus == restricting an evaluation of the FULL corpus (independent recomputation) ----
def full_vs_subset(pc: dict, split: str = "TUNE") -> dict:
    ep = set(SPLIT_IDS[split].tolist())
    bb_full = run_pipeline(MAIN_DET, MAIN, pc).reset_index(drop=True)
    res_full = evaluate(MAIN, bb_full)                                   # the FULL corpus, its own ground-truth cache
    pi = res_full["per_instance"]; pi = pi[pi["episode_id"].isin(ep)].sort_values("track_id").reset_index(drop=True)
    in_t = bb_full["episode_id"].isin(ep).to_numpy()
    bb_sub = run_pipeline(DETS[split], CORP[split], pc)
    res_sub = evaluate(CORP[split], bb_sub)                              # the SUBSET corpus, a fresh cache
    pis = res_sub["per_instance"].sort_values("track_id").reset_index(drop=True)
    loss = episode_losses(CORP[split], DETS[split], pc)                  # the per-episode reduction used by the grid
    return dict(
        same_instances=bool((pi["track_id"].to_numpy() == pis["track_id"].to_numpy()).all()),
        same_miss=bool((pi["instance_missed"].to_numpy() == pis["instance_missed"].to_numpy()).all()
                       and (pi["n_uncovered_frames"].to_numpy() == pis["n_uncovered_frames"].to_numpy()).all()),
        same_boxes=int(in_t.sum()) == len(bb_sub) and int(res_full["blur_is_false"][in_t].sum()) == int(res_sub["blur_is_false"].sum()),
        same_reduction=(int(loss["n_miss"].sum()) == int(pis["instance_missed"].sum()) and int(loss["n_inst"].sum()) == len(pis)
                        and int(loss["n_box"].sum()) == len(bb_sub) and int(loss["n_false"].sum()) == int(res_sub["blur_is_false"].sum())),
        n_inst=len(pis), recall_sub=res_sub["instance_recall"], recall_full_restricted=float(1 - pi["instance_missed"].mean()))

t0 = time.time()
SUBSET_CHECK = [full_vs_subset(pc_of(k)) for k in [(0.7, 1, 30, 20, 0.3), (0.6, 2, 10, 5, 0.1)]]
print(f"subset-vs-full evaluation, 2 configs on TUNE: {time.time() - t0:.1f}s")
for k, r in zip(["(thr .7, stride 1, gap 30, pad 20, margin .3)", "(thr .6, stride 2, gap 10, pad 5, margin .1)"], SUBSET_CHECK):
    print(f"  {k}: n_inst={r['n_inst']} recall subset={r['recall_sub']:.4f} vs full-restricted={r['recall_full_restricted']:.4f}")
check("1.5", "evaluate() on a subset corpus (fresh cache) equals the restriction of the full-corpus evaluation, for 2 configs: identical per-instance "
      "miss flags and uncovered-frame counts, identical box/false-box counts, and identical per-episode reductions (exact equality)",
      all(r["same_instances"] and r["same_miss"] and r["same_boxes"] and r["same_reduction"] for r in SUBSET_CHECK),
      f"{[(r['same_instances'], r['same_miss'], r['same_boxes'], r['same_reduction']) for r in SUBSET_CHECK]}")
# free the full-corpus objects (memory on a 7 GB shared VM): only the splits (own caches) and the TRUTH chunks are used from here on
del MAIN_DET, TRUTH_DET, MAIN, TRUTH
import gc; gc.collect()
print(f"pool densities: {n_inst_main / SPLIT_DF['simulated_minutes'].sum():.3f} face instances per simulated minute "
      f"(natural density; used in section 2 for the staged-capture contrast)")
