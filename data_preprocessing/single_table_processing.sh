#!/bin/bash

python merge_icl_batches_to_h5.py --input-dir ../data_generation/single_table_datasets/single_table_stage1 --output ../model_pretrain/pretrain_datasets/single_table_stage1.h5 --seq-len 600 --num-features 18 --max-num-classes 2 --max-category-cardinality 10

python merge_icl_batches_to_h5.py --input-dir ../data_generation/single_table_datasets/single_table_stage2 --output ../model_pretrain/pretrain_datasets/single_table_stage2.h5 --seq-len 600 --num-features 30 --max-num-classes 2 --max-category-cardinality 10
