# VIZ
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
names = list(H); short = ["(i) unchanged", "(ii) model\nchange", "(iii-a) thr\n0.70->0.90", "(iii-b) thr\n0.70->0.60", "(iv) small\nset (40 ep)"]
vcol = {"PASS": C_LTT, "FAIL": C_REQ, "INCONCLUSIVE": C_CRC}
ax = axes[0]
for i, nm in enumerate(names):
    v = H[nm]
    ax.errorbar(i, v["miss_rate"], yerr=[[v["miss_rate"] - v["miss_rate_lower_bound"]], [v["prop26_U"] - v["miss_rate"]]], fmt="o", color=vcol[v["verdict_ltt_targets"]], capsize=6, ms=9, lw=2)
    ax.text(i + 0.12, v["prop26_U"], v["verdict_ltt_targets"], fontsize=8.5, color=vcol[v["verdict_ltt_targets"]], va="bottom")
ax.axhline(HARNESS_TARGETS["miss"], color=C_LTT, ls="--", lw=1.4); ax.text(len(names) - 0.6, HARNESS_TARGETS["miss"] * 1.05, f"LTT alpha_miss = {HARNESS_TARGETS['miss']}", color=C_LTT, ha="right", fontsize=8.5)
ax.axhline(0.01, color=C_REQ, ls="--", lw=1.4); ax.text(len(names) - 0.6, 0.0105, "client 1% miss", color=C_REQ, ha="right", fontsize=8.5)
ax.set_yscale("log"); ax.set_ylim(0.004, 1.0); ax.set_xticks(range(len(names))); ax.set_xticklabels(short, fontsize=8.5)
ax.set_ylabel("instance miss rate (point, 95% CP interval)"); ax.set_title("reverify(): miss rate against the targets (colour = LTT-target verdict)")
ax = axes[1]
for i, nm in enumerate(names):
    v = H[nm]; lo, hi = v["pooled_precision_boot95"]
    ax.errorbar(i, v["pooled_precision"], yerr=[[v["pooled_precision"] - lo], [hi - v["pooled_precision"]]], fmt="s", color=vcol[v["verdict_client_claims"]], capsize=6, ms=8, lw=2)
    ax.text(i + 0.12, hi, "client: " + v["verdict_client_claims"], fontsize=8.5, color=vcol[v["verdict_client_claims"]], va="bottom")
ax.axhline(0.95, color=C_REQ, ls="--", lw=1.4); ax.text(len(names) - 0.6, 0.955, "client precision 0.95", color=C_REQ, ha="right", fontsize=8.5)
ax.set_ylim(0, 1.05); ax.set_xticks(range(len(names))); ax.set_xticklabels(short, fontsize=8.5)
ax.set_ylabel("pooled precision, cluster-bootstrap 95% interval"); ax.set_title("pooled precision (a bootstrap, not a paper guarantee); colour = verdict on the client's 99%/95% claims")
plt.tight_layout(); plt.show()
