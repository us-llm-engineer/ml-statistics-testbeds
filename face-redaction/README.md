# face-redaction — "Face redaction for egocentric household video"

Upwork job ~022099965157819865153. The client licenses head-mounted (4K, ~30 fps) video of household
cleaning to robotics / embodied-AI companies and needs an unattended CLI that blurs every face before
footage leaves their systems: **recall ≥ 99% per face instance, precision ≥ 95%, minimum $/video-hour**.
They cannot measure recall today because their own sample contains no faces, and they say the answer
to "how will you prove 99%?" matters more than anything else.

This project is a **research-grounded starting point** for a profile project, not a finished product.
The walkthrough notebooks are draft scaffolds on synthetic data with known ground truth, plus one small
real CPU smoke test; they end with explicit limitations and open engineering choices lists.

The statistics follow one story, *calibrate → decide → certify*:

- **Decide (foundational):** Learn then Test — Angelopoulos, Bates, Candès, Jordan, Lei,
  *Annals of Applied Statistics* 19(2), 2025, arXiv:2110.01052. Picks a detector configuration
  (threshold, temporal padding, frame stride, box margin) that holds recall *and* precision targets
  jointly with probability ≥ 1−δ, then the cheapest such configuration.
- **Calibrate detection losses:** Conformal Object Detection by Sequential Risk Control — Andéol,
  Mossina, Mazoyer, Gerchinovitz, arXiv:2505.24038 (2025). Box-level recall losses, matching and
  multiplicative margins; recall control in expectation; precision explicitly not guaranteed.
- **Certify:** Finite-Sample Coverage Audits for High-Recall Candidate Generation — Anthony,
  Salehzadeh Nobari, arXiv:2607.21480 (2026). Exact certificates, the proof that auditing only where
  the detector fired certifies nothing, label-complexity bounds, and stress-test certificates.

Runners-up (not used): Conformal Risk Control (2208.02814, one risk, expectation only); Active
Statistical Inference (2403.03208) and Prediction-Powered Active Testing (2607.08347) (asymptotic
intervals, weak at 99% with few positives); SUPG (2004.00827, older than 5 years); EgoBlur
(2308.13093, domain baseline without theory — cited as external context only).

Contents:
- `nlm/`: NotebookLM queue (alias `facered`, profile `second-plus`, 3 PDFs, 12 gated answers in
  `responses/facered-q*.json`; `bootstrap.sh` and the profile-patched hook scripts).
- `research/face-redaction-research-log.md`: all 12 answers with design notes.
- `research/notebook-research-context.md`: consistency contract (lookup only) and section → query map.
- `research/reread.sh`: prints selected answers from the live NotebookLM history (local fallback).
- `notebooks/01_research_foundations.ipynb`: the math, with Monte Carlo checks of each guarantee.
- `notebooks/02_project_walkthrough_part1.ipynb`: synthetic egocentric episodes, per-frame baseline,
  tracking/padding, a real YuNet + ffmpeg smoke test, and a baseline audit.
- `notebooks/03_project_walkthrough_part2.ipynb`: applies the three papers to notebook 2's pipeline
  (it executes notebook 2's `# REUSE` cells verbatim): a larger corpus with disjoint TUNE / LTT-CAL /
  sealed-AUDIT episode sets, staged-capture sizing, Learn then Test configuration choice (client
  targets T1 and a relaxed demonstration target T2, verified over 200 re-splits), a SeqCRC contrast,
  audit certificates (Prop. 26, excluded-pool, Thm 19, Thm 29), cost per option, a re-verification
  harness, a results summary, and the limitations / open engineering choices / what the sources do
  not settle lists. Built from `tools/nb3_cells/*.py` (code) and `tools/nb3_md/*.md` (narrative) by
  `tools/build_nb3.py`; `data/nb3/` holds the pre-registration and re-verification JSON files.
- `data/weights/face_detection_yunet_2023mar.onnx`: OpenCV Zoo YuNet (MIT licence, Shiqi Yu),
  sha256 `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`.
- `tools/run_notebook.py`: executes a notebook on the `face-redaction` kernel and checks every
  `# VIZ` cell produced an image.

Environment: `.venv` (Python 3.12; numpy, scipy, matplotlib, pandas, opencv-python-headless 5.0,
scikit-image, imageio-ffmpeg 7.0.2 static, nbclient), Jupyter kernel `face-redaction`.
No LLM calls: the job does not need one.

## Key results (all synthetic unless labelled; see the notebooks for the checks behind each)

- Notebook 1: every guarantee verified by simulation; the audit paper's numbers reproduce (sample-size
  table, worked example). One printed value in that paper (0.0116) is a rounding slip (exact 0.01152);
  the notebook keeps that FAIL visible.
- Notebook 2: a naive per-frame baseline reaches 0.113 strict instance recall (0.905 frame recall); no
  configuration in the 108-config grid meets recall >= 0.99 and precision >= 0.95 under either
  screen/photo policy. Pets dominate false blurs only because of a generator setting.
- Notebook 3: the client's targets (miss <= 1%, false-blur <= 5%) give **no certified configuration**.
  A relaxed target, revised once after a failed first attempt (disclosed, tested on a fresh split),
  certifies two stride-1 configurations. On the sealed audit the client's 99% recall claim is **not**
  certified (recall bound 0.967 for the chosen config, 0.988 for the best-effort one, at pooled
  precision about 0.55 and 0.24). Four printed checks are honest FAILs: CRC's violation rate (0.117)
  did not reach the pre-stated 0.152 (three checks) and one audited instance breaks Thm 19's nesting.
- Costs are measured CPU throughput times **assumed** prices; measured smoke timings varied about
  two-fold between runs on a shared WSL2 VM.

## Rebuilding and re-running

    .venv/bin/python tools/build_nb3.py            # assemble notebook 3 from tools/nb3_cells + tools/nb3_md
    .venv/bin/python tools/run_notebook.py notebooks/03_project_walkthrough_part2.ipynb 3000
    bash research/reread.sh 7 8                    # re-read NotebookLM answers behind a section

Notebooks 1 and 2 are rebuilt with `tools/build_nb1.py` and `tools/build_nb2.py`. Runtime on a shared
12-core VM: about 1 min, 3.5 min and 6 min for notebooks 1, 2 and 3.
