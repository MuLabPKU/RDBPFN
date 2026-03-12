# Data Preprocessing

This directory converts generated data and benchmark datasets into the formats required for pretraining and evaluation.

It supports two workflows:

1. single-table preprocessing
2. relational-database preprocessing

## Installation

This stage is packaged through [pyproject.toml](pyproject.toml).

From the repository root:

```bash
cd data_preprocessing
pip install -e .
```

The DFS preprocessing pipeline is partially adapted from the [dbinfer](https://github.com/awslabs/multi-table-benchmark) project. You can also refer to that repository for additional usage details. We gratefully acknowledge their work.

## Directory Highlights

- [single_table_processing.sh](single_table_processing.sh): merges generated single-table batches into `.h5` priors.
- [RDB_processing.sh](RDB_processing.sh): end-to-end relational preprocessing pipeline.
- [merge_icl_batches_to_h5.py](merge_icl_batches_to_h5.py): merges single-table generation outputs into `.h5`.
- [merge_dbinfer_to_h5.py](merge_dbinfer_to_h5.py): converts processed RDB tasks into `.h5`.
- [filter_h5_sampling_columns.py](filter_h5_sampling_columns.py): downsamples columns from unsampled `.h5` files.

## Workflow 1: Single-Table Preprocessing

### Purpose

This workflow takes raw synthetic single-table batches and merges them into pretraining-ready `.h5` datasets.

### Input

Expected default inputs:

- `../data_generation/single_table_datasets/single_table_stage1`
- `../data_generation/single_table_datasets/single_table_stage2`

These are produced by [../data_generation/single_table/single_table_generate.sh](../data_generation/single_table/single_table_generate.sh).

### Output

Default outputs:

- `../model_pretrain/pretrain_datasets/single_table_stage1.h5`
- `../model_pretrain/pretrain_datasets/single_table_stage2.h5`

### Usage

```bash
cd data_preprocessing
bash single_table_processing.sh
```

### What the Script Does

- reads synthetic batch directories
- merges them into `.h5`
- writes pretraining-ready files into `model_pretrain/pretrain_datasets/`

## Workflow 2: RDB Preprocessing

### Purpose

This workflow converts raw synthetic RDBs into:

- processed task directories produced by the DFS-based preprocessing pipeline.
- intermediate unsampled `.h5` files
- final sampled `.h5` files used for RDB_PFN pretraining

### Input

Expected default inputs are the raw RDB directories generated under:

- `../data_generation/RDB_datasets/`

### Output

Default outputs include:

- intermediate `.h5` files under `RDB_datasets/`
- sampled pretraining `.h5` files under `../model_pretrain/pretrain_datasets/`

### Usage

```bash
cd data_preprocessing
bash RDB_processing.sh
```

### What the Script Does

The current pipeline performs two stages:

1. It runs DFS-based preprocessing on each raw RDB directory.
2. It converts the processed outputs into `.h5`, then downsamples columns into final pretraining datasets.

Note the DFS preprocessing can take hundreds of hours to complete. You can modify the script to run on a subset of the datasets for testing.

## Handoff to Model Pretraining

After preprocessing:

1. Use `model_pretrain/pretrain_datasets/` as the training corpus for pretraining.
2. Use benchmark-ready dataset directories under `model_pretrain/rdb_datasets/` for evaluation.
3. Continue with [../model_pretrain/README.md](../model_pretrain/README.md).
