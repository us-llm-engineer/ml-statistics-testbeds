# ---- Two target sets, FIXED BEFORE any CAL or AUDIT label is touched (P3 Assumption 1 discipline; recorded in an event log + JSON) ----
# T1 = the client's targets (given). T2-v1 = a relaxed DEMONSTRATION target, chosen by the following rule using TUNE point estimates ONLY:
#   for alpha_miss in the ladder [0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20] (increasing), let S = grid configs whose TUNE miss rate is
#   <= alpha_miss - 0.02 (a 2-point headroom, ~2 binomial SE at n=437). Take the first alpha_miss with |S| >= 8; then
#   alpha_fb = the smallest multiple of 0.05 that is >= (min TUNE episode-FDP over S) + 0.05 (a 5-point headroom, ~3 SE at n=650).
def choose_t2(tune_miss, tune_fb, ladder=(0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20), headroom_miss=0.02, headroom_fb=0.05, min_configs=8):
    for a in ladder:
        S = np.where(tune_miss <= a - headroom_miss)[0]
        if len(S) >= min_configs:
            return dict(miss=a, fb=float(np.round(np.ceil((tune_fb[S].min() + headroom_fb) / 0.05 - 1e-9) * 0.05, 2)), n_configs=len(S))
    return None
_t2 = choose_t2(S_TUNE["miss"], S_TUNE["fb"])
ALPHA_T2_V1 = dict(miss=_t2["miss"], fb=_t2["fb"])          # first pre-registered T2 (revised in section 3(c) after a failed first test)
PREREG = dict(delta=DELTA, T1=ALPHA_T1, T2_v1=ALPHA_T2_V1, T2_rule="see cell comment: smallest alpha_miss on the ladder with >= 8 configs at TUNE miss <= alpha-0.02; alpha_fb = ceil_to_0.05(min TUNE FDP over those + 0.05)",
              tune_configs_within_headroom=_t2["n_configs"], fixed_at=time.strftime("%Y-%m-%d %H:%M:%S"))
log_event("targets T1/T2 fixed from TUNE only (CAL not yet evaluated)")
(NB3_DIR / "prereg_targets.json").write_text(json.dumps(PREREG, indent=2))
print(json.dumps(PREREG, indent=2))
print(f"\nT1 (client):  alpha_miss = {ALPHA_T1['miss']}, alpha_fb = {ALPHA_T1['fb']}, delta = {DELTA}")
print(f"T2-v1 (relaxed demonstration, from the TUNE frontier): alpha_miss = {ALPHA_T2_V1['miss']}, alpha_fb = {ALPHA_T2_V1['fb']}, delta = {DELTA}   "
      f"[{_t2['n_configs']} configs sit within the headroom on TUNE]")
check("3.1", "T2-v1 was fixed by the written rule from TUNE alone: the event log shows the targets were recorded before the CAL grid was evaluated, and T2-v1 is "
      "strictly looser than T1 on both risks", ALPHA_T2_V1["miss"] > ALPHA_T1["miss"] and ALPHA_T2_V1["fb"] > ALPHA_T1["fb"] and EVENTS[-1][1].startswith("targets"),
      f"T2-v1 = ({ALPHA_T2_V1['miss']}, {ALPHA_T2_V1['fb']}); events so far: {[e[1][:28] for e in EVENTS]}")
