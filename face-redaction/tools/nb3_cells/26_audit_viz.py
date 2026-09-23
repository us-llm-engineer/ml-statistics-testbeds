# VIZ
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
ax = axes[0]
Qc = np.arange(30, 1400)
for C_, ls_, col in ((0, "-", C_CERT), (1, "--", C_CERT), (2, ":", C_CERT)):
    ax.plot(Qc, 1 - cp_upper(C_, Qc, DELTA_AUDIT), ls_, color=col, label=f"P26 bound if C = {C_} misses")
ax.axhline(0.99, color=C_REQ, lw=1.4, ls="--"); ax.text(1390, 0.9915, "client: 99%", color=C_REQ, ha="right", fontsize=9)
for name, mk, col in (("T2", "o", C_LTT), ("BEST", "s", C_CRC)):
    a = AUD[name]
    ax.scatter(a["Q"], 1 - a["U"], marker=mk, s=110, color=col, ec=PALETTE["ink"], zorder=5, label=f"AUDIT {name}: Q={a['Q']}, C={a['C']} -> {1 - a['U']:.3f}")
ax.set_ylim(0.85, 1.003); ax.set_xlabel("audited face instances Q"); ax.set_ylabel("certified recall lower bound 1 - U_Q(C, 0.05)")
ax.set_title("Prop. 26: certified recall vs Q"); ax.legend(fontsize=8, loc="lower right")

ax = axes[1]
thr = T19["thr"].to_numpy()
ax.plot(thr, T19["instance_miss_rate"], "o-", color=C_LTT, label="audit miss rate (instances)")
ax.plot(thr, T19["U_Q_C_inst"], "s--", color=C_CERT, label="Prop. 26 upper bound U_Q(C, 0.05)")
ax.axhline(EPS_INST, color=C_REQ, ls="--", lw=1.2); ax.text(0.405, EPS_INST + 0.006, f"eps_inst = {EPS_INST} (twin of Thm 19)", color=C_REQ, fontsize=8.5)
ax.axhline(0.01, color=PALETTE["stone"], ls=":", lw=1.2); ax.text(0.405, 0.013, "client 1%", color=PALETTE["stone"], fontsize=8.5)
ax.plot(thr, T19["r_bound_seg"], "^-", color=PALETTE["sand"], label="Thm 8 missed-MASS bound r (45-s segments)")
if ret_inst >= 0:
    ax.scatter([NESTED_THR[ret_inst]], [T19.loc[ret_inst, "U_Q_C_inst"]], s=220, marker="*", color=C_LTT, ec=PALETTE["ink"], zorder=6, label=f"instance-level walk returns thr {NESTED_THR[ret_inst]}")
ax.set_yscale("symlog", linthresh=0.01); ax.set_ylim(0, 0.5); ax.set_xlabel("score threshold (nested family, 0.40 = most inclusive)"); ax.set_ylabel("rate / bound")
ax.set_title("Thm 19 walk: segment mass passes everywhere, instances do not"); ax.legend(fontsize=8, loc="upper left")

ax = axes[2]
lab, val, col = [], [], []
for name, c_ in (("T2", C_LTT), ("BEST", C_CRC)):
    a = AUD[name]; t = T29[T29["config"] == name].iloc[0]
    for l_, v_ in ((f"{name}: P26 miss-rate bound", a["U"]), (f"{name}: rho_avg (stress)", t["rho_avg_bound"]), (f"{name}: rho_max (stress)", t["rho_max_bound"])):
        lab.append(l_); val.append(v_); col.append(c_)
lab.append("T2: missed mass r (segments, Thm 8)"); val.append(r_hyper); col.append(PALETTE["sand"])
y = np.arange(len(lab))[::-1]
ax.barh(y, val, color=col, ec=PALETTE["ink"])
for yi, v_ in zip(y, val):
    ax.text(v_ * 1.06, yi, f"{v_:.3f}", va="center", fontsize=8.5)
ax.axvline(0.01, color=C_REQ, ls="--", lw=1.4); ax.text(0.0105, y.max() + 0.55, "1%", color=C_REQ, fontsize=9)
ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=8.5); ax.set_xscale("log"); ax.set_xlim(0.005, 1.0)
ax.set_xlabel("certified UPPER bound (log)"); ax.set_title("Per-certificate summary (delta = 0.05; Thm 29 at delta/(2|G|))")
plt.tight_layout(); plt.show()
