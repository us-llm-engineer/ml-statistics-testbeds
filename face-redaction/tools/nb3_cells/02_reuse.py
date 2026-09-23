# Execute Notebook 2's `# REUSE` cells VERBATIM, in order, in this notebook's namespace.
# (SimConfig, generate_corpus, simulate_detector, run_pipeline, evaluate, apply_policy,
#  false_blur_decomposition, processing_cost, ...). Nothing of NB2 is re-typed below.
import hashlib
import re as _re

_NB2 = nbformat.read(NB2_PATH, as_version=4)
_REUSE_SRC = [c.source for c in _NB2.cells if c.cell_type == "code" and c.source.startswith("# REUSE")]
for _i, _src in enumerate(_REUSE_SRC):
    exec(compile(_src, f"nb2_reuse_cell_{_i}", "exec"), globals())
print(f"executed {len(_REUSE_SRC)} NB2 REUSE cells verbatim; sha1 of each source:")
for _i, _src in enumerate(_REUSE_SRC):
    print(f"  reuse[{_i}] {hashlib.sha1(_src.encode()).hexdigest()[:10]}  {len(_src):5d} chars  first def: "
          f"{(_re.findall(r'^(?:def|class)\s+(\w+)', _src, flags=_re.M) or ['(constants)'])[0]}")
check("0.1", "NB2 has exactly 6 '# REUSE' cells and all executed (a-priori: count == 6)",
      len(_REUSE_SRC) == 6, f"{len(_REUSE_SRC)} cells")

# NB2's own printed numbers (stored outputs of its cells) are the reference for reproduction.
def _nb2_outputs_containing(text):
    for c in _NB2.cells:
        if c.cell_type == "code":
            for o in c.get("outputs", []):
                if o.get("output_type") == "stream" and text in o.get("text", ""):
                    return o["text"]
    return ""
_o6 = _nb2_outputs_containing("generate_corpus:")
_o15 = _nb2_outputs_containing("CHECK 2.1")
NB2_REF = dict(
    n_episodes=int(_re.search(r"(\d+) episodes", _o6).group(1)),
    n_tracks=int(_re.search(r"(\d+) tracks", _o6).group(1)),
    naive_recall=float(_re.search(r"CHECK 2\.1.*?-> ([0-9.]+) ->", _o15).group(1)),
    naive_frame_recall=float(_re.search(r"CHECK 2\.2.*?-> ([0-9.]+)", _o15).group(1)),
    naive_precision=float(_re.search(r"CHECK 2\.3.*?-> ([0-9.]+) ->", _o15).group(1)),
    n_instances=int(_re.search(r"for (\d+) instances", _o15).group(1)),
)
print("NB2 printed reference (parsed from its stored outputs):", NB2_REF)

NB2_CORPUS = generate_corpus(SimConfig(seed=SEED))            # exactly NB2's corpus: SimConfig(seed=SEED)
NB2_DET = simulate_detector(NB2_CORPUS, {}, seed=SEED)
_NAIVE_PC = dict(score_thr=0.5, stride=1, max_gap=0, pad_before=0, pad_after=0, margin=0.0)
_nres = evaluate(NB2_CORPUS, run_pipeline(NB2_DET, NB2_CORPUS, _NAIVE_PC))
check("0.2", "generate_corpus(SimConfig(seed=SEED)) reproduces NB2's episode and track counts exactly",
      (len(NB2_CORPUS["episodes"]), len(NB2_CORPUS["tracks"])) == (NB2_REF["n_episodes"], NB2_REF["n_tracks"]),
      f"{len(NB2_CORPUS['episodes'])} episodes, {len(NB2_CORPUS['tracks'])} tracks (NB2 printed {NB2_REF['n_episodes']}, {NB2_REF['n_tracks']})")
check("0.3", "evaluate() on NB2's naive config reproduces NB2's printed 3-decimal instance recall, frame recall, "
      "precision and instance count (a-priori: |diff| <= 5e-4)",
      abs(_nres["instance_recall"] - NB2_REF["naive_recall"]) <= 5e-4 and abs(_nres["frame_recall"] - NB2_REF["naive_frame_recall"]) <= 5e-4
      and abs(_nres["precision"] - NB2_REF["naive_precision"]) <= 5e-4 and _nres["n_instances"] == NB2_REF["n_instances"],
      f"recall {_nres['instance_recall']:.3f}/{NB2_REF['naive_recall']}, frame {_nres['frame_recall']:.3f}/{NB2_REF['naive_frame_recall']}, "
      f"precision {_nres['precision']:.3f}/{NB2_REF['naive_precision']}, n={_nres['n_instances']}/{NB2_REF['n_instances']}")
_d0 = generate_corpus(SimConfig())
print(f"note: SimConfig() with the default seed=0 gives {len(_d0['tracks'])} tracks and "
      f"{int(_d0['tracks']['is_face_to_redact'].sum())} face instances; NB2's own corpus uses seed=SEED "
      f"({len(NB2_CORPUS['tracks'])} tracks, {NB2_REF['n_instances']} instances).")
del NB2_DET, _nres, _d0
