# VIZ
fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
fine = "fine (step 0.01)"
D_ = PROC_DF[PROC_DF["family"].isin([fine, "full grid"])].set_index("procedure")
order = ["CRC (instances)", "CRC (episode-averaged)", "LTT-1D (miss only)", "LTT-full T2 (both risks; section 3)"]
cols = [C_CRC, PALETTE["berry"], C_LTT, PALETTE["slate"]]; short = ["CRC\n(instances)", "CRC\n(episode-avg)", "LTT-1D\n(miss only)", "LTT-full\n(T2)"]
tolc_ = 3 * np.sqrt(DELTA * (1 - DELTA) / R_CRC)

ax = axes[0, 0]
for proc, c_, mk in (("CRC", C_CRC, "o"), ("LTT-1D", C_LTT, "s")):
    d_ = SWEEP_DF[SWEEP_DF["procedure"] == proc]
    ax.plot(d_["alpha"], d_["violation_rate"], mk + "-", color=c_, label=proc, ms=7)
ax.axhline(DELTA, color=C_REQ, ls="--", lw=1.5); ax.text(0.0705, DELTA + 0.01, "delta = 0.10", color=C_REQ, ha="right", fontsize=9)
ax.set_xlabel("miss target alpha"); ax.set_ylabel("fraction of calibration sets with TRUE miss > alpha")
ax.set_title(f"Violation vs alpha (calibration set = {N_CAL} episodes)"); ax.set_ylim(0, 0.5); ax.legend()

ax = axes[0, 1]
for proc, c_, mk in (("CRC", C_CRC, "o"), ("LTT-1D", C_LTT, "s")):
    d_ = SWEEP_N_DF[SWEEP_N_DF["procedure"] == proc]
    ax.plot(d_["cal_instances_approx"], d_["violation_rate"], mk + "-", color=c_, label=proc, ms=7)
    if proc == "LTT-1D":
        ax.plot(d_["cal_instances_approx"], d_["abstain_rate"], ":", color=C_LTT, label="LTT-1D abstain rate")
ax.axhline(DELTA, color=C_REQ, ls="--", lw=1.5)
ax.set_xscale("log"); ax.set_xlabel(f"face instances in the calibration set (approx.), alpha = {ALPHA_CRC}"); ax.set_ylabel("fraction of calibration sets")
ax.set_title("Small calibration sets: CRC violates more, LTT abstains more"); ax.set_ylim(0, 1.0); ax.legend(fontsize=8.5)

ax = axes[1, 0]
m = [D_.loc[o, "mean_true_miss"] for o in order]
ax.bar(short, m, color=cols, ec=PALETTE["ink"])
ax.axhline(ALPHA_CRC, color=C_REQ, ls="--", lw=1.5); ax.text(3.45, ALPHA_CRC + 0.003, f"alpha = {ALPHA_CRC}", color=C_REQ, ha="right", fontsize=9)
for i, x_ in enumerate(m):
    ax.text(i, x_ + 0.002, f"{x_:.3f}", ha="center", fontsize=10)
ax.set_ylabel("mean TRUE miss risk of the selected config"); ax.set_title("Expectation holds for CRC; face-free episodes dilute an episode-averaged loss")

ax = axes[1, 1]
pp = [D_.loc[o, "mean_true_pooled_precision"] for o in order]
ax.bar(short, pp, color=cols, ec=PALETTE["ink"])
ax.axhline(0.95, color=C_REQ, ls="--", lw=1.5); ax.text(3.45, 0.965, "client: precision 0.95", color=C_REQ, ha="right", fontsize=9)
for i, x_ in enumerate(pp):
    ax.text(i, x_ + 0.012, f"{x_:.2f}", ha="center", fontsize=10)
ax.set_ylim(0, 1.05); ax.set_ylabel("mean TRUE pooled precision of the selected config")
ax.set_title("Precision: not controlled by SeqCRC (P2 Sec. V-D); nobody here reaches 0.95")
plt.tight_layout(); plt.show()
