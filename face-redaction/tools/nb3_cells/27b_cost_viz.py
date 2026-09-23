# VIZ
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
ax = axes[0]
strd = np.array([k[1] for k in ALL_KEYS]); x = np.zeros(G)
for s_ in (1, 2, 4):                                    # strip layout: within a stride, configs are spread left-to-right by increasing CAL2 miss
    m_ = np.where(strd == s_)[0]; order_ = m_[np.argsort(p2["miss"][m_])]
    x[order_] = s_ + np.linspace(-0.4, 0.4, len(order_)) * (0.55 if s_ > 1 else 1.0)
unc = ~lam2
sc = ax.scatter(x[unc], COST_VEC[unc], c=np.clip(p2["miss"][unc], 0, 0.15), cmap="Oranges", s=55, ec=PALETTE["stone"], vmin=0, vmax=0.15, label="not certified (colour = CAL2 miss)")
ax.scatter(x[lam2], COST_VEC[lam2], marker="*", s=260, color=C_LTT, ec=PALETTE["ink"], zorder=5, label="certified (T2)")
ax.scatter([x[J_BEST]], [COST_VEC[J_BEST]], marker="D", s=80, color=C_CRC, ec=PALETTE["ink"], zorder=4, label="best-effort recall-first (uncertified)")
plt.colorbar(sc, ax=ax, label="CAL2 miss rate")
ax.set_xticks([1, 2, 4]); ax.set_xlabel("detector stride (run YuNet every s-th frame)"); ax.set_ylabel("CPU $/video-hour (MEASURED throughput x ASSUMED price)")
ax.set_title("Cost vs stride: certified vs uncertified configs"); ax.legend(fontsize=8, loc="upper right")

ax = axes[1]
sp = np.linspace(1, 32, 200)
for gp, col in ((0.30, C_CERT), (0.60, C_LTT), (1.20, C_CRC)):
    ax.plot(sp, COST_BY_STRIDE[1]["cpu_compute_s_per_video_s"] / sp * gp, color=col, label=f"GPU at ${gp}/h (ASSUMED)")
ax.axhline(COST_BY_STRIDE[1]["cpu_usd_per_video_hour"], color=C_REQ, ls="--", lw=1.5); ax.text(31, COST_BY_STRIDE[1]["cpu_usd_per_video_hour"] + 0.05, f"CPU: ${COST_BY_STRIDE[1]['cpu_usd_per_video_hour']:.2f}", color=C_REQ, ha="right", fontsize=9)
ax.axvline(BE_SPEEDUP, color=PALETTE["stone"], ls=":"); ax.text(BE_SPEEDUP + 0.3, 0.25, f"break-even {BE_SPEEDUP:.0f}x\n(at \\$0.60 vs \\$0.05 per hour)", fontsize=8.5, color=PALETTE["slate"])
ax.axvline(ASSUMED_GPU_SPEEDUP, color=PALETTE["amber"], ls=":"); ax.text(ASSUMED_GPU_SPEEDUP + 0.3, 3.0, f"assumed {ASSUMED_GPU_SPEEDUP:.0f}x", fontsize=8.5, color=PALETTE["amber"])
ax.set_ylim(0, 4.2); ax.set_xlabel("ASSUMED GPU speed-up over the measured CPU path"); ax.set_ylabel("$/video-hour (stride 1)"); ax.set_title("GPU vs CPU break-even (all GPU numbers ASSUMED)"); ax.legend(fontsize=8)

ax = axes[2]
w = 0.36; xs = np.arange(len(VAR))
for k_, (s, col) in enumerate(((1, C_LTT), (4, C_CRC))):
    vals = VAR_DF[VAR_DF["stride"] == s]["cpu_usd_per_video_hour"].to_numpy()
    ax.bar(xs + (k_ - 0.5) * w, vals, w, color=col, ec=PALETTE["ink"], label=f"stride {s}")
    for xi, v_ in zip(xs, vals):
        ax.text(xi + (k_ - 0.5) * w, v_ + 0.1, f"{v_:.2f}", ha="center", fontsize=8.5)
ax.set_xticks(xs); ax.set_xticklabels(["slow smoke run", "as measured\nin timings.json", "fast smoke run"], fontsize=9)
ax.set_ylabel("CPU USD per video-hour (ASSUMED price 0.05 USD/h)"); ax.set_title("Run-to-run variance of the measured smoke timings"); ax.legend()
plt.tight_layout(); plt.show()
