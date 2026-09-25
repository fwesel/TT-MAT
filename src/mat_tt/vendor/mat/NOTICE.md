# Vendored code notice

The code in this directory is adapted from the
[Molecule Attention Transformer (MAT)](https://github.com/ardigen/MAT) repository
by Maziarka et al., *"Molecule Attention Transformer"* (2020), https://arxiv.org/abs/2002.08264.

Upstream is MIT-licensed by Ardigen (see `LICENSE` in this directory, copied unmodified).

## transformer.py

The MAT model definition, copied verbatim except that the bare `from utils import ...` was made
relative (`from .utils import ...`) so this works as an installed package instead of requiring
`os.chdir('src')` the way `MAT/EXAMPLE.ipynb` does. A provenance docstring was added at the top.
The architecture itself is untouched.

## utils.py

Upstream imports three private symbols from `torch.nn.init` at module scope
(`_calculate_fan_in_and_fan_out`, `_no_grad_normal_`, `_no_grad_uniform_`). They are private and
have changed signature across torch releases, and an import failure there would take the whole model
definition down with it. `xavier_normal_small_init_` and `xavier_uniform_small_init_` are therefore
rewritten on public APIs, keeping the same fan computation (verified to match torch's private helper
exactly) and the same resulting distribution.

Upstream's `earily_stop` is not vendored. It tracks an accuracy-style higher-is-better metric, which
does not fit RMSE, so `mat_tt.train` implements its own early stopping.

## featurization/data_utils.py

Only the SMILES to featurized-graph functions are kept: `load_data_from_df`,
`load_data_from_smiles`, `featurize_mol`, `get_atom_features`, `one_hot_vector`, `pad_array`.

`mat_tt.data` calls `featurize_mol` (and through it `get_atom_features`) directly, and reimplements
only the conformer-generation step of `load_data_from_smiles` with the same procedure (add H,
`EmbedMolecule(maxAttempts=5000)`, UFF-optimize, remove H, else 2D coordinates), checking
`EmbedMolecule`'s return code so a 3D failure is reported rather than silently substituted with 2D
coordinates. `load_data_from_df`/`load_data_from_smiles` are retained unmodified for reference.
`pad_array` is used by `mat_tt.torch_data` so batching keeps MAT's padding semantics.

The bare `except:` around the 3D-embedding fallback in `load_data_from_smiles` was narrowed to
`except Exception:` (flagged by ruff's `E722`), behavior is unchanged.

## What is deliberately not vendored

`Molecule`, `MolDataset`, `mol_collate_func`, `construct_dataset` and `construct_loader`. They build
`torch.cuda.FloatTensor` when CUDA is present, which hardcodes the device, and they return tensors
in an order (adjacency, features, distance, labels) that is easy to mis-unpack.
`mat_tt.torch_data` provides a device-agnostic, named equivalent built on the same `pad_array`.
