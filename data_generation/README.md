# Data Generation

This directory contains the data generation stage of RDB_PFN. It has two subprojects:

1. `single_table/` for synthetic single-table priors.
2. `RDB/` for synthetic relational database generation.

The outputs of both subprojects feed into the preprocessing stage documented in [../data_preprocessing/README.md](../data_preprocessing/README.md).

## Directory Layout

`single_table/`
Python package for generating synthetic single-table tasks.

`RDB/`
Python package for generating synthetic RDBs.

`single_table_datasets/`
Default output location for generated single-table batches.

`RDB_datasets/`
Default output location for generated synthetic relational databases.

## Subproject 1: Single-Table Generation

### Purpose

This subproject generates large synthetic single-table batches that are later merged into `.h5` priors for model pretraining.

### Installation

From the repository root:

```bash
cd data_generation/single_table
pip install -e .
```

The single-table generation code is adapted from the [tabicl](https://github.com/tabicl/tabicl) project with slight modifications to the prior. You can also refer to that repository for additional usage details. We gratefully acknowledge their work.


### Main Entry Points

- [single_table/single_table_generate.sh](single_table/single_table_generate.sh): launches the default generation runs.
- `src/tabicl/prior/genload.py`: lower-level generator invoked by the shell script.

### Default Usage

Run the provided generation script:

```bash
cd data_generation/single_table
bash single_table_generate.sh
```

The current script generates:

- `single_table_datasets/single_table_stage1`
- `single_table_datasets/single_table_stage2`

These directories are consumed later by [../data_preprocessing/single_table_processing.sh](../data_preprocessing/single_table_processing.sh).

### Notes

- The provided script is configured for large-scale generation, which may take tens of hours to complete.
- Generation parameters such as `--num_batches`, feature count, class count, and sequence length are currently hard-coded in the shell script. You can modify them to generate smaller datasets for testing.

## Subproject 2: RDB Generation

### Purpose

This subproject generates synthetic relational databases with multiple variants.

### Installation

From the repository root:

```bash
cd data_generation/RDB
pip install -e .
```

Because this codebase does not rely on complex packaging, it is usually straightforward to run it in another environment as long as PyTorch and the required dependencies are installed.


### Main Entry Points

- [RDB/RDB_generate.sh](RDB/RDB_generate.sh): launches the default RDB generation schedule.
- [RDB/dag_to_rdb_generator.py](RDB/dag_to_rdb_generator.py): main generator script.

### Default Usage

Run the provided generation script:

```bash
cd data_generation/RDB
bash RDB_generate.sh
```

The current script creates multiple raw synthetic datasets under `RDB_datasets/`, including:

- small and large prior configurations
- variants with and without GNN-based generation
- pre-split parts for later preprocessing with different DFS hop settings

These outputs are consumed later by [../data_preprocessing/RDB_processing.sh](../data_preprocessing/RDB_processing.sh).

### Notes

- Output directories and dataset counts are currently hard-coded in the shell script.
- The `--use_row_gnn` flag controls whether row-level graph structure is used during generation.
- The current default script is large-scale and may require substantial runtime (could take days) and storage. You can modify the script to generate smaller datasets or more aggressive parallelization for testing.

## Handoff to Preprocessing

After generation, use [../data_preprocessing/README.md](../data_preprocessing/README.md) for converting single-table batches into `.h5` priors and raw RDBs into processed task datasets and pretraining files.
