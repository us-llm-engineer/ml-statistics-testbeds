# Provenance: where each notebook was executed

Each notebook below was executed once, on the machine named, and saved with its outputs. Nothing was re-executed locally after being copied back, and no
output cell was edited. Markdown cells were edited after execution to quote the printed numbers; the code cells and their outputs are as executed.

| Notebook | Executed on | Wall time | Environment |
|---|---|---|---|
| `01_research_foundations.ipynb` | Modal CPU sandbox, 16 cores | 22 s | Python 3, numpy 2.5.3, scipy 1.18.1, pandas 3.0.5, matplotlib 3.11.2 |
| `02_mock_project_part1_build.ipynb` | local workstation, 12 cores | about 70 s (unchanged since the first version) | numpy 2.5.3, scipy 1.18.1 |
| `03_mock_project_part2_prove_screen_combine.ipynb` | Modal CPU sandbox, 12 cores (`NB3_WORKERS=12`) | 38 s (the 200-world audit took 16 s) | same pins as `requirements.txt` |
| `04_paper_reproductions.ipynb` | Modal CPU sandbox, 12 cores | 81 s | same pins as `requirements.txt` |

Notes.

- Notebook 1 and Notebook 3 were re-executed after fixing the confidence sequence's tuning constant; Notebook 2 does not use it. Notebook 3 ran four times in
  total on the cloud: with the fixed constant and the old `v_opt` (2 cores, Colab, 941 s; superseded), with `v_opt = 10` and the 95% monitor (12 cores, Modal; AC4 revised
  horizon 0.480, superseded), with `v_opt = 10` and the one-sided 5% monitor (12 cores, Modal; superseded by the next run only in that it lacks section 6b), and the same plus the section 6b frontier of fixed-checkpoint z-tests (12 cores, Modal; the committed run). The ordering and the reasons are in section 6 of the notebook.
- Design constants were compared on prototype worlds (seeds 900-959, disjoint from the audit seeds 1000-1199) with scripts that ran locally on 3 worker processes for
  a few minutes each; those scripts are not part of the repository and their outputs are summarised in the notebook text only.
- Executed notebooks were copied back as gzip and base64 with a sha256 check on both ends. The Modal container reports 28 to 32 logical CPUs from `os.cpu_count()`
  but was allocated 12 or 16 cores.
- Notebook 4 downloads `forecasts/mlb_2010_2019.csv` from the authors' public repository (`yjchoe/ComparingForecasters`, MIT licence) at commit
  `52748c86e0429a9612dc79892c3be6156a524132` at run time. The file is third-party data and is not part of this repository.
