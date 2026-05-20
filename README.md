# PDH-Diffusion

Official codebase for the ICLR 2025 paper:
**Synthesizing Realistic fMRI: A Physiological Dynamics-Driven Hierarchical Diffusion Model for Efficient fMRI Acquisition**.

This repository contains the cleaned and reproducible training/evaluation pipeline for the HCP fMRI forecasting setup used in our experiments.

## Highlights

- Multi-granularity diffusion forecasting with graph/hypergraph guidance.
- Reproducibility-focused script with fixed stable defaults.
- Ready-to-use environment exports (`environment_pdhd.yml` and `requirements_pdhd.txt`).
- Minimal code layout for open-source release and maintenance.

## Repository Structure

```text
.
├── data/                               # Local HCP data (see data/README.md)
├── log/                                # Training/evaluation logs (generated)
├── result/                             # Exported prediction results (generated)
├── scripts/
│   └── run_pdhd_gran3.sh               # Main reproducible multi-run script
├── src/
│   ├── run_pdhd_hcp.py                 # Main HCP training/evaluation entrypoint
│   ├── pdhd_estimator.py               # Estimator wrapper
│   ├── pdhd_network_Mamba.py           # Model network
│   ├── pdhd_module.py                  # Diffusion module
│   ├── trainer.py                      # Training loop
│   ├── multi_gran_generator.py         # Multi-granularity data/graph builder
│   ├── metrics.py                      # Evaluation metrics
│   ├── data_provider/                  # Data loading utilities
│   └── hypergraph/                     # Hypergraph layers
├── environment_pdhd.yml                # Conda environment export
└── requirements_pdhd.txt               # pip requirements export
```

## 1) Environment Setup

We provide two ways to reproduce the environment.

### Option A (Recommended): Conda YAML

```bash
conda env create -f environment_pdhd.yml -n pdhd
conda activate pdhd
```

> Note: `environment_pdhd.yml` was exported from a local machine and may contain a `prefix` field.
> Using `-n pdhd` ensures the environment is created under your local conda path.

### Option B: pip Requirements

```bash
conda create -n pdhd python=3.9.12 -y
conda activate pdhd
pip install -r requirements_pdhd.txt
```

## 2) Dataset Preparation (HCP)

The training script reads preprocessed JSON files from a **data root directory**.

According to the paper setup, the HCP split is:

- Train samples: 696
- Test samples: 174
- Each sample shape: `NROI x T`, with `NROI=82`, `T=1200`

### Data root resolution

The data root is resolved in this order:

1. Environment variable `PDHD_DATA_ROOT` (if set)
2. Project default: `./data` (relative to the repository root)

### Download (Hugging Face)

The preprocessed HCP JSON files (~1.28 GB total) are hosted on Hugging Face:

**https://huggingface.co/datasets/Yvnnone/formatted_hcp**

Files in the dataset:

| File | Description | Approx. size |
|------|-------------|--------------|
| `formatted_data_corr_HCP_train.json` | Training set (696 subjects) | ~974 MB |
| `formatted_data_corr_HCP_test.json` | Test set (174 subjects) | ~244 MB |

Each record contains:

- `target`: ROI time series, shape `82 x 1200`
- `corr`: ROI correlation matrix, shape `82 x 82`
- `start`: series start timestamp

Download into the project `./data` folder (recommended):

```bash
pip install -U huggingface_hub

huggingface-cli download Yvnnone/formatted_hcp \
  --repo-type dataset \
  --local-dir data \
  --include "formatted_data_corr_HCP_train.json" "formatted_data_corr_HCP_test.json"
```

Verify:

```bash
ls data/formatted_data_corr_HCP_train.json data/formatted_data_corr_HCP_test.json
```

> These files are **not** committed to this GitHub repository (see `.gitignore`).

### Optional cached files (auto-generated)

On the first run, the script may create and reuse processed caches under the same data root:

- `processed_d_data_HCP_train_<mg_dict>_<graph_percentage>.json`
- `processed_d_data_HCP_test_<mg_dict>_<graph_percentage>.json`

For the default script settings (`mg_dict=1_4_8`, `graph_percentage=0.95`), these caches are large (~3–4 GB) and **do not need to be downloaded**; they will be built locally from the two `formatted_*` files above.

### Setup example

**Option A — default `./data` folder:**

```bash
# download from Hugging Face (see above), then:
bash scripts/run_pdhd_gran3.sh
```

**Option B — custom data directory:**

```bash
export PDHD_DATA_ROOT="/path/to/your/data"
huggingface-cli download Yvnnone/formatted_hcp \
  --repo-type dataset \
  --local-dir "${PDHD_DATA_ROOT}" \
  --include "formatted_data_corr_HCP_train.json" "formatted_data_corr_HCP_test.json"
bash scripts/run_pdhd_gran3.sh
```

## 3) Run Experiments

### One-command reproducible run (recommended)

```bash
bash scripts/run_pdhd_gran3.sh
```

### Common overrides

```bash
CONDA_ENV_NAME=pdhd \
GPU_IDS="0 1 2 3 4 5 6 7" \
NUM_REPS=5 \
EPOCH=80 \
BATCH_SIZE=96 \
LEARNING_RATE=7e-6 \
MG_DICT=1_4_8 \
SHARE_RATIO_LIST=1_0.1_0.1 \
WEIGHT_LIST=0.8_0.1_0.1 \
bash scripts/run_pdhd_gran3.sh
```

## 4) Key Hyperparameters

In `scripts/run_pdhd_gran3.sh`:

- `MG_DICT`: granularity levels (e.g., `1_4_8` for 1h/4h/8h targets).
- `NUM_GRAN`: number of granularities; must match `MG_DICT`.
- `SHARE_RATIO_LIST`: diffusion-step sharing ratio across granularities.
- `WEIGHT_LIST`: final forecasting loss weights across granularities.
- `LOSS_WEIGHT_LIST`: internal objective weighting  
  (`diffusion_loss_weight`, `fractal_loss_weight`).
- `GRAPH_PERCENTAGE`: ratio used in graph edge construction/filtering.
- `NUM_REPS`: repeated runs for stability/variance evaluation.

## 5) Outputs

- Logs: `log/<model_name>_<dataset>/`
- Results: `result/<model_name>_<dataset>/`

For the default script:

- `model_name=pdhd`
- `dataset=hcp`

so outputs are in:

- `log/pdhd_hcp/`
- `result/pdhd_hcp/`

## 6) Reproducibility Notes

- Use fixed seeds and identical hardware/software stack whenever possible.
- Keep `NUM_REPS > 1` to report mean/std rather than a single run.
- Prefer `environment_pdhd.yml` over generic pip installation for closest reproduction.
- Random seed defaults to `2020` in `src/run_pdhd_hcp.py` (`--seed`, `--eval_seed`).
  The `SEED` variable in `scripts/run_pdhd_gran3.sh` is passed to the Python entrypoint.

## 7) Troubleshooting

- **Conda env not found**:  
  set `CONDA_ENV_NAME` explicitly when launching the script.
- **Data file missing**:  
  download from [Yvnnone/formatted_hcp](https://huggingface.co/datasets/Yvnnone/formatted_hcp) into `data/`, or set `PDHD_DATA_ROOT` correctly.
- **GPU memory issue**:  
  lower `BATCH_SIZE` first (e.g., 96 -> 64 or 48).
- **Long training time**:  
  reduce `EPOCH` / `NUM_REPS`, or increase available GPUs.

## 8) Citation

Please cite the PDHDiffusion paper if you use this codebase.

```bibtex
@inproceedings{hu2025synthesizing,
  title={Synthesizing Realistic fMRI: A Physiological Dynamics-Driven Hierarchical Diffusion Model for Efficient fMRI Acquisition},
  author={Hu, Yufan and Jiang, Yu and Li, Wuyang and Yuan, Yixuan},
  booktitle={International Conference on Learning Representations},
  year={2025}
}
```

## 9) License

This project is released under the license specified in `LICENSE`.
