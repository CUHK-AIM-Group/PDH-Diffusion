# HCP Data Directory

Place the preprocessed HCP JSON files in this folder before training.

## Download

Dataset: **[Yvnnone/formatted_hcp](https://huggingface.co/datasets/Yvnnone/formatted_hcp)** on Hugging Face (~1.28 GB).

```bash
pip install -U huggingface_hub

huggingface-cli download Yvnnone/formatted_hcp \
  --repo-type dataset \
  --local-dir data \
  --include "formatted_data_corr_HCP_train.json" "formatted_data_corr_HCP_test.json"
```

## Required files

| File | Description |
|------|-------------|
| `formatted_data_corr_HCP_train.json` | 696 training subjects (~974 MB) |
| `formatted_data_corr_HCP_test.json` | 174 test subjects (~244 MB) |

Verify:

```bash
ls formatted_data_corr_HCP_train.json formatted_data_corr_HCP_test.json
```

These files are **not** tracked in git (see root `.gitignore`).

## Optional cache (auto-generated)

The training script may create large processed caches here on first run, e.g.:

- `processed_d_data_HCP_train_[1.0, 4.0, 8.0]_0.95.json`
- `processed_d_data_HCP_test_[1.0, 4.0, 8.0]_0.95.json`

You do not need to download these manually.
