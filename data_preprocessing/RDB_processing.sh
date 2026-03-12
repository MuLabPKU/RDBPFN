#!/bin/bash

# Deep Feature Synthesis converting raw RDBs to single-table tasks
bash benchmark_preprocess_depth1.sh ../data_generation/RDB_datasets/RDB_small_woGNN_1hop_1_raw 8
bash benchmark_preprocess_depth1.sh ../data_generation/RDB_datasets/RDB_small_wGNN_1hop_1_raw 8
bash benchmark_preprocess_depth1.sh ../data_generation/RDB_datasets/RDB_small_woGNN_1hop_2_raw 8
bash benchmark_preprocess_depth1.sh ../data_generation/RDB_datasets/RDB_small_wGNN_1hop_2_raw 8
bash benchmark_preprocess_depth1.sh ../data_generation/RDB_datasets/RDB_small_woGNN_2hop_1_raw 8
bash benchmark_preprocess_depth1.sh ../data_generation/RDB_datasets/RDB_small_wGNN_2hop_1_raw 8
bash benchmark_preprocess_depth1.sh ../data_generation/RDB_datasets/RDB_large_woGNN_1hop_sub1_raw 8
bash benchmark_preprocess_depth1.sh ../data_generation/RDB_datasets/RDB_large_woGNN_1hop_sub2_raw 8
bash benchmark_preprocess_depth1.sh ../data_generation/RDB_datasets/RDB_large_wGNN_1hop_sub1_raw 8
bash benchmark_preprocess_depth1.sh ../data_generation/RDB_datasets/RDB_large_wGNN_1hop_sub2_raw 8

# Converting single-table tasks to h5 format and do downsampling
python merge_dbinfer_to_h5.py  --dataset-root ../data_generation/RDB_datasets/RDB_small_woGNN_1hop_1_raw-processed  --output RDB_datasets/RDB_small_woGNN_1hop_1_unsampled.h5  --total-rows 600 --max-columns 60 --min-train-ratio 0.5 --max-train-ratio 0.9
python filter_h5_sampling_columns.py RDB_datasets/RDB_small_woGNN_1hop_1_unsampled.h5 ../model_pretrain/pretrain_datasets/RDB_small_woGNN_1hop_1.h5  --sampled-columns 30  --max-expected-columns 10  --ratio 0.9 --safety-factor 1.0

python merge_dbinfer_to_h5.py  --dataset-root ../data_generation/RDB_datasets/RDB_small_woGNN_1hop_2_raw-processed  --output RDB_datasets/RDB_small_woGNN_1hop_2_unsampled.h5  --total-rows 600 --max-columns 60 --min-train-ratio 0.5 --max-train-ratio 0.9
python filter_h5_sampling_columns.py RDB_datasets/RDB_small_woGNN_1hop_2_unsampled.h5 ../model_pretrain/pretrain_datasets/RDB_small_woGNN_1hop_2.h5  --sampled-columns 30  --max-expected-columns 10  --ratio 0.9 --safety-factor 1.0

python merge_dbinfer_to_h5.py  --dataset-root ../data_generation/RDB_datasets/RDB_small_wGNN_1hop_1_raw-processed  --output RDB_datasets/RDB_small_wGNN_1hop_1_unsampled.h5  --total-rows 600 --max-columns 60 --min-train-ratio 0.5 --max-train-ratio 0.9
python filter_h5_sampling_columns.py RDB_datasets/RDB_small_wGNN_1hop_1_unsampled.h5 ../model_pretrain/pretrain_datasets/RDB_small_wGNN_1hop_1.h5  --sampled-columns 30  --max-expected-columns 10  --ratio 0.9 --safety-factor 1.0

python merge_dbinfer_to_h5.py  --dataset-root ../data_generation/RDB_datasets/RDB_small_wGNN_1hop_2_raw-processed  --output RDB_datasets/RDB_small_wGNN_1hop_2_unsampled.h5  --total-rows 600 --max-columns 60 --min-train-ratio 0.5 --max-train-ratio 0.9
python filter_h5_sampling_columns.py RDB_datasets/RDB_small_wGNN_1hop_2_unsampled.h5 ../model_pretrain/pretrain_datasets/RDB_small_wGNN_1hop_2.h5  --sampled-columns 30  --max-expected-columns 10  --ratio 0.9 --safety-factor 1.0

python merge_dbinfer_to_h5.py  --dataset-root ../data_generation/RDB_datasets/RDB_small_woGNN_2hop_1_raw-processed  --output RDB_datasets/RDB_small_woGNN_2hop_1_unsampled.h5  --total-rows 600 --max-columns 90 --min-train-ratio 0.5 --max-train-ratio 0.9
python filter_h5_sampling_columns.py RDB_datasets/RDB_small_woGNN_2hop_1_unsampled.h5 ../model_pretrain/pretrain_datasets/RDB_small_woGNN_2hop_1.h5  --sampled-columns 30  --max-expected-columns 10  --ratio 0.9 --safety-factor 1.0

python merge_dbinfer_to_h5.py  --dataset-root ../data_generation/RDB_datasets/RDB_small_wGNN_2hop_1_raw-processed  --output RDB_datasets/RDB_small_wGNN_2hop_1_unsampled.h5  --total-rows 600 --max-columns 90 --min-train-ratio 0.5 --max-train-ratio 0.9
python filter_h5_sampling_columns.py RDB_datasets/RDB_small_wGNN_2hop_1_unsampled.h5 ../model_pretrain/pretrain_datasets/RDB_small_wGNN_2hop_1.h5  --sampled-columns 30  --max-expected-columns 10  --ratio 0.9 --safety-factor 1.0

python merge_dbinfer_to_h5.py  --dataset-root ../data_generation/RDB_datasets/RDB_large_woGNN_1hop_sub1_raw-processed --dataset-root ../data_generation/RDB_datasets/RDB_large_woGNN_1hop_sub2_raw-processed  --output RDB_datasets/RDB_large_woGNN_1hop_1_unsampled.h5  --total-rows 600 --max-columns 60 --min-train-ratio 0.5 --max-train-ratio 0.9
python filter_h5_sampling_columns.py RDB_datasets/RDB_large_woGNN_1hop_1_unsampled.h5 ../model_pretrain/pretrain_datasets/RDB_large_woGNN_1hop_1.h5  --sampled-columns 30  --max-expected-columns 10  --ratio 0.9 --safety-factor 1.0

python merge_dbinfer_to_h5.py  --dataset-root ../data_generation/RDB_datasets/RDB_large_wGNN_1hop_sub1_raw-processed --dataset-root ../data_generation/RDB_datasets/RDB_large_wGNN_1hop_sub2_raw-processed  --output RDB_datasets/RDB_large_wGNN_1hop_1_unsampled.h5  --total-rows 600 --max-columns 60 --min-train-ratio 0.5 --max-train-ratio 0.9
python filter_h5_sampling_columns.py RDB_datasets/RDB_large_wGNN_1hop_1_unsampled.h5 ../model_pretrain/pretrain_datasets/RDB_large_wGNN_1hop_1.h5  --sampled-columns 30  --max-expected-columns 10  --ratio 0.9 --safety-factor 1.0