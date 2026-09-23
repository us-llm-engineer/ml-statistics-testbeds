# VIZ
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

ax = axes[0]
cs = np.arange(0, 9)
for delta, ls in [(0.05, "-"), (0.10, "--")]:
    ax.plot(cs, [min_q(C, EPS_TARGET, delta) for C in cs], "o" + ls, color=C_CERT, label=f"Prop. 26, delta={delta}")
ax.axhline(Q_AUDIT, color=C_REQ, lw=1.6)
ax.text(8, Q_AUDIT + 25, f"sealed AUDIT: Q={Q_AUDIT}", ha="right", color=C_REQ, fontsize=9)
ax.axhline(299, color=PALETTE["stone"], lw=1, ls=":"); ax.text(8, 299 + 25, "299 (P3 Sec. 4.6, C=0)", ha="right", color=PALETTE["stone"], fontsize=8)
ax.set_xlabel("misses C allowed among the Q audited face instances"); ax.set_ylabel("audited face instances Q")
ax.set_title("Q needed to certify miss rate <= 1%"); ax.legend(loc="upper left")

ax = axes[1]
qs = np.arange(10, 1300, 10)
ax.plot(qs, qs / INSTANCES_PER_MINUTE_STAGED / 60, color=C_LTT, label=f"staged, ASSUMED {INSTANCES_PER_MINUTE_STAGED:g} instance/min")
ax.plot(qs, qs / NATURAL_DENSITY_PER_MIN / 60, color=PALETTE["sand"], label=f"natural density, SYNTHETIC {NATURAL_DENSITY_PER_MIN:.2f}/min")
for C, mk in zip([0, 1, 2], ["o", "s", "^"]):
    q = int(SIZING_DF.loc[SIZING_DF["allowed_misses_C"] == C, "Q_min_delta0.05"].iloc[0])
    ax.plot(q, q / INSTANCES_PER_MINUTE_STAGED / 60, mk, color=C_CERT, ms=8, label=f"C={C}: Q={q}")
ax.set_yscale("log"); ax.set_xlabel("face instances to capture (Q)"); ax.set_ylabel("footage hours (log)")
ax.set_title("footage needed: staged vs natural"); ax.legend(loc="upper left")

ax = axes[2]
x = np.arange(len(COMP_DF)); w = 0.38
ax.bar(x - w / 2, COMP_DF["natural_share"], w, color=PALETTE["sand"], label="natural share (corpus)")
ax.bar(x + w / 2, COMP_DF["staged_share"], w, color=[KIND_COLORS[k] for k in COMP_DF["kind"]], label="recommended staged share")
for i, r in COMP_DF.iterrows():
    ax.text(i + w / 2, r["staged_share"] + 0.006, f"{r['tune_miss_rate']:.2f}", ha="center", fontsize=8, color=C_REQ)
ax.set_xticks(x); ax.set_xticklabels([k.replace("_", "\n") for k in COMP_DF["kind"]], fontsize=8)
ax.set_ylabel("share of audited instances"); ax.set_title("what to shoot (red = TUNE miss rate of the kind)"); ax.legend(loc="upper right")
plt.tight_layout(); plt.show()
