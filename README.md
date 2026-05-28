# CaCuTe Experiments

This repository contains research code and notebooks for the paper
[CaCuTe: Casual Cubic-Model Technique for Faster Optimization](https://arxiv.org/abs/2509.18508).

The codebase is based on
[konstmish/opt_methods](https://github.com/konstmish/opt_methods),
which provides the original package structure, losses, tracing utilities,
and a large part of the baseline optimization code. This repository is a
research fork/adaptation focused on the CaCuTe experiments and related method
variants.

## What Is In This Repo

- `optmethods/`: optimization methods, losses, dataset loaders, and trace utilities
- `optmethods/second_order/`: CaCuTe-related second-order and hybrid methods
- `notebooks/`: experiment notebooks used to run and compare methods
- `optmethods/datasets/`: local datasets used by the notebooks

Main experiment notebooks:

- `notebooks/1.CaCuN.ipynb`
- `notebooks/2.CaCuAdaN.ipynb`
- `notebooks/3.CaCuAdGD.ipynb`
- `notebooks/3.1.CaCuAdGD.ipynb` - post-rebuttal update of the CaCuAdGD experiments
- `notebooks/4.CaCuSGD.ipynb`

## Environment

For the runs that were reproduced in this repository, the closest working
environment was:

- Python `3.10.12`
- `numpy==2.2.6`
- `scipy==1.15.3`
- `scikit-learn==1.7.1`
- `matplotlib==3.10.5`
- `seaborn==0.13.2`

A practical setup with `uv`:

```bash
uv python install 3.10.12
uv venv --python 3.10.12 .venv310
source .venv310/bin/activate
uv pip install numpy==2.2.6 scipy==1.15.3 scikit-learn==1.7.1 matplotlib==3.10.5 seaborn==0.13.2 jupyter ipykernel
uv pip install -e .
```

Note: `requirements.txt` currently contains `sklearn`, but for fresh installs
you should use `scikit-learn`.

## Running The Notebooks

From the repository root:

```bash
source .venv310/bin/activate
jupyter lab
```

Then open the notebooks from `notebooks/`.

The notebooks use `notebooks/jupyter_utils.py` to add the repository root to
`sys.path`, so they are intended to be run as notebooks, not as plain scripts.

## Datasets

Dataset loading is implemented in `optmethods/datasets/utils.py`.

- Local LIBSVM-style files are read from `optmethods/datasets/`
- Some datasets are loaded through `scikit-learn` fetchers
- This repository already includes several local datasets such as
  `w8a`, `a1a`, `a5a`, `a9a`, `mushrooms`, `covtype.bz2`, and `news20.bz2`

If you add new local datasets, place them in `optmethods/datasets/` and load
them through `optmethods.datasets.get_dataset(...)` or directly in a notebook.

## Notes

- This is research code, not a polished library release
- The notebooks are the main entry point for reproducing figures and comparisons
- Some package metadata in `setup.py` still points to the upstream
  `konstmish/opt_methods` repository because this code started from that base

## References

- Paper: [CaCuTe: Casual Cubic-Model Technique for Faster Optimization](https://arxiv.org/abs/2509.18508)
- DOI: [10.1145/3770855.3817869](https://doi.org/10.1145/3770855.3817869)
- Base repository: [konstmish/opt_methods](https://github.com/konstmish/opt_methods)
