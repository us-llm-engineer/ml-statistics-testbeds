# ---- LTT building blocks, re-implemented compactly here (nothing imported from Notebook 1) ----
DELTA = 0.10                                   # error level; P1's default delta = 0.10 [facered-q2/q10]
ALPHA_T1 = dict(miss=0.01, fb=0.05)            # T1: the CLIENT's targets (recall >= 99% per instance; false-blur risk <= 5%)

def h1(a, b):
    a = np.clip(np.asarray(a, float), 0, 1)
    t1 = np.where(a > 0, a * np.log(np.maximum(a, 1e-300) / b), 0.0)
    t2 = np.where(a < 1, (1 - a) * np.log(np.maximum(1 - a, 1e-300) / (1 - b)), 0.0)
    return t1 + t2

def p_binom(k, n, alpha):
    """Exact binomial-tail p-value for a BINARY loss with k losses in n trials [P1 Sec. 3.2, facered-q2]: P(Bin(n, alpha) <= k)."""
    return stats.binom.cdf(k, n, alpha)

def p_hb(rhat, n, alpha):
    """Hoeffding-Bentkus p-value for a loss bounded in [0,1] [P1 Prop. 1, Eq. 1, facered-q2]."""
    rhat = np.asarray(rhat, float)
    bentkus = np.e * stats.binom.cdf(np.ceil(n * rhat - 1e-9), n, alpha)
    hoeffding = np.exp(-n * h1(np.minimum(rhat, alpha), alpha))
    return np.minimum(np.minimum(hoeffding, bentkus), 1.0)

def fixed_sequence(p_path, delta, starts=(0,)):
    """Algorithm 1 [P1 Sec. 2.3.1, facered-q2]: walk each start forward along the path while p <= delta/|J|; stop at the first failure.
    Returns (accepted mask over path positions, visited mask = positions whose p-value had to be evaluated)."""
    L = len(p_path); lvl = delta / len(starts)
    acc, vis = np.zeros(L, bool), np.zeros(L, bool)
    for s in starts:
        j = s
        if acc[j]:
            continue                                   # 'if lambda_j not in Lambda-hat'
        while j < L:
            vis[j] = True
            if p_path[j] <= lvl:
                acc[j] = True; j += 1
            else:
                break
    return acc, vis

# ---- losses from cached per-episode counts. Two risks, with EXPLICIT units (both derived here; only the LTT framework is P1) ----
#  R_miss(lam) = P(a face-to-redact INSTANCE has >= 1 uncovered visible frame): binary loss per instance; empirical = pooled misses / instances
#  R_fb(lam)   = E_episode[ FDP_e ],  FDP_e = (false blur box-frames)/(blur box-frames) of episode e, defined as 0 when the episode draws no box
#               (an FDR-style risk in [0,1] per EPISODE [P1 Sec. 3.1 FDR construction]); NOT the same as pooled precision (see section 5(e))
def to_matrices(loss_by_key: dict, work_names: list, keys: list) -> dict:
    """Stack per-config per-episode counts (from eval_many) over the given work chunks into matrices [G, E]."""
    def stack(field):
        return np.stack([np.concatenate([loss_by_key[(w, k)][field] for w in work_names]) for k in keys])
    n_inst = np.concatenate([loss_by_key[(w, keys[0])]["n_inst"] for w in work_names])
    n_miss, n_box, n_false = stack("n_miss"), stack("n_box"), stack("n_false")
    fdp = np.where(n_box > 0, n_false / np.maximum(n_box, 1), 0.0)
    return dict(n_inst=n_inst, n_miss=n_miss, n_box=n_box, n_false=n_false, fdp=fdp)

def risk_summary(M: dict, idx) -> dict:
    """Point estimates + counts on the episodes `idx` (vector over configs): miss rate, episode-averaged FDP, pooled precision."""
    ni = M["n_inst"][idx].sum(); ne = len(idx)
    return dict(n_inst=int(ni), n_ep=ne, miss=M["n_miss"][:, idx].sum(1) / ni, fb=M["fdp"][:, idx].mean(1),
                pooled_prec=1 - M["n_false"][:, idx].sum(1) / np.maximum(M["n_box"][:, idx].sum(1), 1),
                k_miss=M["n_miss"][:, idx].sum(1))

def ltt_pvalues(M: dict, idx, alpha: dict) -> dict:
    """p-values of both risks for ALL configs on the episodes idx, and the Prop. 6 max [P1 Prop. 6, facered-q2]."""
    s = risk_summary(M, idx)
    p_m = p_binom(s["k_miss"], s["n_inst"], alpha["miss"])
    p_f = p_hb(s["fb"], s["n_ep"], alpha["fb"])
    return dict(p_miss=p_m, p_fb=p_f, p_max=np.maximum(p_m, p_f), **{k: s[k] for k in ("miss", "fb", "pooled_prec", "n_inst", "n_ep")})

def learn_path(p_tune: dict, alpha: dict) -> np.ndarray:
    """Fixed test path learned on TUNE ONLY (P1 App. D split fixed-sequence; the path uses I_graph, the test uses I_testing).
    DERIVED HERE (a variation on App. D): rank ALL grid points by their TUNE max-p (Prop. 6), ties broken by the larger safety headroom
    max(R_miss/alpha_miss, R_fb/alpha_fb); every grid point is reachable, which matters because the cheap (strided) configs must be testable."""
    excess = np.maximum(p_tune["miss"] / alpha["miss"], p_tune["fb"] / alpha["fb"])
    return np.lexsort((excess, p_tune["p_max"]))

def appD_beta_path(p_tune: dict, alpha: dict, D: int = 200) -> np.ndarray:
    """P1 App. D's own path: lam~(beta) = argmin_j || [p_j1, p_j2] - [beta, beta] ||_inf on the TUNE p-values, beta = d/D; adjacent duplicates removed."""
    P = np.stack([p_tune["p_miss"], p_tune["p_fb"]], axis=1)
    path = [int(np.argmin(np.max(np.abs(P - d / D), axis=1))) for d in range(D + 1)]
    return np.array([j for i, j in enumerate(path) if i == 0 or j != path[i - 1]])

def ltt_select(M_tune, tune_idx, M_test, test_idx, alpha, delta=DELTA, starts=(0,)):
    """Whole LTT procedure of one split: path on TUNE -> fixed-sequence test on the test episodes -> (Lambda-hat mask over configs, details)."""
    pt = ltt_pvalues(M_tune, tune_idx, alpha)
    path = learn_path(pt, alpha)
    pc_ = ltt_pvalues(M_test, test_idx, alpha)
    acc_path, vis_path = fixed_sequence(pc_["p_max"][path], delta, starts)
    lam_hat = np.zeros(len(path), bool); lam_hat[path[acc_path]] = True
    visited = np.zeros(len(path), bool); visited[path[vis_path]] = True
    return lam_hat, dict(path=path, p_tune=pt, p_test=pc_, visited=visited)
