# EMolES

EMolES is a density-matrix electronic-structure model library for electrolyte
descriptors. It predicts one-electron reduced density matrices in an LCAO
operator representation so downstream descriptors such as HOMO/LUMO levels,
orbital isosurfaces, electrostatic potential surfaces, dipoles, local charge,
and Li+ solvation-shell deformation are derived from a shared electronic
structure source.

The model stack follows the EMolES paper definition:

- EAMP blocks for electrostatic-aware message passing on 3D atomic graphs.
- Bond-aligned SO(2) layers for local directional contacts such as Li-O,
  Li-F, S-O, and C-O.
- Edge-wise tensor-product self-mix for directional charge redistribution in
  multi-donor and anion-solvent environments.
- Element-conditioned tensor-product output layers for density-matrix node and
  edge blocks.

This repository is an independent EMolES runtime. A small `dptb` compatibility
shim is kept only for transitional checkpoint/module path resolution.

## Install

Install PyTorch for your CUDA or CPU environment first, then install EMolES:

```bash
git clone https://github.com/Franklalalala/EMolES.git
cd EMolES
pip install .
```

The training runtime depends on `e3nn`, `lmdb`, `ase`, `h5py`, `torch-runstats`,
and the usual scientific Python stack declared in `pyproject.toml`.

## Train

EMolES keeps the minimal commands needed to reproduce model training:

```bash
emoles config input.json --train --e3tb
emoles train input.json --output runs/emoles
```

An editable starter config is included at
`examples/minimal_train/input_emoles.json`; update its `data_options.train.root`
and `prefix` to point at the LMDB training set.

The retained code path covers dataset construction, EMolES model construction,
training, validation monitors, and checkpoint saving. Legacy CLI workflows for
band plotting, standalone inference, SK conversion, NRL conversion, and broad
test suites were removed from this repository.

## Checkpoints And EMolStudio

Paper-era checkpoints are not vendored in this repository. The manuscript
workspace checkpoint manifest used during migration is:

```text
E:/thu/emolkit_paper/0512_data/emoles_test_results_fixed/checkpoints/checkpoint_manifest.json
```

EMolStudio should import the new package through `emoles`; existing adapter code
can forward historical checkpoint paths to this runtime during migration.

## Links

- EMolES repository: https://github.com/Franklalalala/EMolES
