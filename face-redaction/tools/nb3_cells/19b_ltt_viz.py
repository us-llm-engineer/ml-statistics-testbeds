# VIZ
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
p2 = LTT_MAIN["T2"]["p_test"]; lam2 = LTT_MAIN["T2"]["lam_hat"]

ax = axes[0, 0]
ax.add_patch(plt.Rectangle((0, 0), ALPHA_T1["fb"], ALPHA_T1["miss"], fill=False, ec=C_REQ, lw=2, ls="--"))
ax.add_patch(plt.Rectangle((0, 0), ALPHA_T2["fb"], ALPHA_T2["miss"], fill=False, ec=C_LTT, lw=2, ls="--"))
ax.annotate("T1 (client)\nbox", (ALPHA_T1["fb"], ALPHA_T1["miss"]), xytext=(0.02, 0.03), color=C_REQ, fontsize=9, arrowprops=dict(arrowstyle="-", color=C_REQ))
ax.annotate("T2 target box", (ALPHA_T2["fb"], ALPHA_T2["miss"]), xytext=(0.5, 0.0035), color=C_LTT, fontsize=9, arrowprops=dict(arrowstyle="-", color=C_LTT))
strd = np.array([k[1] for k in ALL_KEYS])
for s_, mk in ((1, "o"), (2, "s"), (4, "^")):
    m = (strd == s_) & ~lam2
    ax.scatter(p2["fb"][m], p2["miss"][m], marker=mk, s=42, color=PALETTE["stone"], alpha=0.75, label=f"stride {s_}, not certified")
ax.scatter(p2["fb"][lam2], p2["miss"][lam2], marker="*", s=260, color=C_LTT, ec=PALETTE["ink"], zorder=5, label="certified (Lambda-hat, T2)")
ax.set_yscale("symlog", linthresh=0.01); ax.set_xlim(0, 0.85); ax.set_ylim(0, 0.6)
ax.set_xlabel("false-blur risk R_fb (episode-averaged FDP), CAL2"); ax.set_ylabel("miss risk R_miss (instances), CAL2")
ax.set_title("Grid of 58 configs, CAL2 point estimates"); ax.legend(loc="upper right", fontsize=8)

ax = axes[0, 1]
x = np.arange(len(PATH))
ax.plot(x, P_TUNE["p_max"][PATH], color=PALETTE["stone"], lw=1.5, label="TUNE max-p (defines the path)")
ax.plot(x, p2["p_max"][PATH], color=C_LTT, lw=1.5, label="CAL2 max-p (tested)")
ax.axhline(DELTA, color=C_REQ, ls="--", lw=1.3); ax.text(len(PATH) - 1, DELTA * 1.25, f"delta = {DELTA}", color=C_REQ, ha="right", fontsize=9)
acc_pos = np.where(lam2[PATH])[0]; vis_pos = np.where(LTT_MAIN["T2"]["visited"][PATH])[0]
ax.scatter(vis_pos, p2["p_max"][PATH][vis_pos], s=70, color="white", ec=PALETTE["ink"], zorder=4, label="p evaluated by the lazy walk")
ax.scatter(acc_pos, p2["p_max"][PATH][acc_pos], s=110, marker="*", color=C_LTT, ec=PALETTE["ink"], zorder=5, label="certified")
ax.set_yscale("log"); ax.set_ylim(1e-6, 2); ax.set_xlabel("position on the path learned from TUNE (0 = looked safest)"); ax.set_ylabel("max p-value over the two risks (Prop. 6)")
ax.set_title("Fixed-sequence test along the TUNE path (T2)"); ax.legend(loc="lower right", fontsize=8)

ax = axes[1, 0]
rng_j = np.random.default_rng(3)
names = ["T2", "T2 multi-start J=3", "T2 multi-start J=5"]
for k, nm in enumerate(names):
    ch = RS[nm]["chosen"]; ch = ch[ch >= 0]
    tm = np.array([TRUTH_CACHE[ALL_KEYS[j]]["miss"] for j in ch])
    ax.scatter(tm, k + rng_j.uniform(-0.25, 0.25, len(tm)), s=14, alpha=0.5, color=C_LTT)
    st_ = RS_STATS.set_index("procedure").loc[nm]
    ax.text(0.075, k, f"abstain {st_['abstain_rate']:.0%}, violated {int(st_['violations'])}/{int(st_['R'])}", va="center", fontsize=8.5)
ax.axvline(ALPHA_T2["miss"], color=C_REQ, ls="--", lw=1.5); ax.text(ALPHA_T2["miss"] + 0.001, 2.45, "alpha_miss", color=C_REQ, fontsize=9)
ax.set_yticks(range(3)); ax.set_yticklabels(["single start", "3 starts", "5 starts"]); ax.set_xlim(0.0, 0.14); ax.set_ylim(-0.6, 2.7)
ax.set_xlabel(f"TRUE miss risk of the selected config ({N_TRUTH}-episode TRUTH population)"); ax.set_title(f"{R_SPLITS} re-splits: true risk of what LTT selected (T2)")

ax = axes[1, 1]
labs = ["L0\nbaseline"] + [f"L{i + 1}" for i in range(len(SENS_ROWS))]
bars = ax.bar(labs, SENS_EXCESS, color=[PALETTE["stone"]] + [C_LTT if e <= 1 else PALETTE["amber"] for e in SENS_EXCESS[1:]], ec=PALETTE["ink"])
ax.axhline(1.0, color=C_REQ, ls="--", lw=1.5); ax.text(3.5, 1.3, "below the line:\nT1 met by point estimate\n(not certified)", color=C_REQ, ha="center", va="bottom", fontsize=8)
for b, e in zip(bars, SENS_EXCESS):
    ax.text(b.get_x() + b.get_width() / 2, e + 0.1, f"{e:.1f}x", ha="center", fontsize=9)
ax.set_ylabel("best config: max(miss/0.01, FDP/0.05), CAL"); ax.set_title("Sensitivity: better detector / policy (T1 certified at NONE of them)")
plt.tight_layout(); plt.show()
