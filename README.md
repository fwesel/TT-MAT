# TT compression of attention in a Molecular Transformer

This case study tests whether Tensor Train (TT) linear layers can reduce the
parameter count of a small Molecule Attention Transformer on ESOL.

## Reproduce

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
uv run jupyter nbconvert --to notebook --execute \
  --output case_study.ipynb notebooks/case_study.ipynb
uv run ruff check .
uv run ruff format --check .
```

The notebook reads committed JSON results and runs in seconds. To retrain:

```bash
uv run python scripts/run_task2.py
uv run python scripts/run_task3.py
uv run python scripts/run_task4.py
uv run python scripts/run_task4_rank_lr.py
uv run python scripts/run_task4_probes.py
uv run python scripts/run_task7_rf.py
uv run python scripts/run_task7_low_rank.py
```

## Files

- `notebooks/case_study.ipynb`: main reproducible notebook and figures
- `src/mat_tt/tt.py`: reusable core-by-core `TTLinear`
- `src/mat_tt/low_rank.py`: low-rank matrix-factorization control layer
- `src/mat_tt/data.py`: canonicalization, splitting, deterministic conformers,
  and cache handling
- `src/mat_tt/train.py`: train-only target scaling, training, and evaluation
- `src/mat_tt/compress.py`: replacement of MAT linear layers with TT layers
- `src/mat_tt/report.py`: tables and primary figures
- `results/`: committed run records used by the notebook

The MAT implementation is vendored under its MIT license in
`src/mat_tt/vendor/mat/`. TensorLy and TensorLy-Torch provide TT-SVD and the TT
container. Python, PyTorch, RDKit, pandas, scikit-learn, Matplotlib, uv, Ruff,
and GitHub Copilot were also used. GitHub Copilot assisted with writing the
experiment code.
