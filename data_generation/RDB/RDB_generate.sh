#! /bin/bash

python dag_to_rdb_generator.py --output_base_dir ../RDB_datasets/RDB_small_woGNN_1hop_1_raw --num_rdbs 80000 --num_processes 1 --start_index 0

python dag_to_rdb_generator.py --output_base_dir ../RDB_datasets/RDB_small_wGNN_1hop_1_raw --use_row_gnn --num_rdbs 80000 --num_processes 1 --start_index 80000 

python dag_to_rdb_generator.py --output_base_dir ../RDB_datasets/RDB_small_woGNN_1hop_2_raw --num_rdbs 80000 --num_processes 1 --start_index 160000

python dag_to_rdb_generator.py --output_base_dir ../RDB_datasets/RDB_small_wGNN_1hop_2_raw --use_row_gnn --num_rdbs 80000 --num_processes 1 --start_index 240000

python dag_to_rdb_generator.py --output_base_dir ../RDB_datasets/RDB_small_woGNN_2hop_1_raw --num_rdbs 40000 --num_processes 1 --start_index 320000

python dag_to_rdb_generator.py --output_base_dir ../RDB_datasets/RDB_small_wGNN_2hop_1_raw --use_row_gnn --num_rdbs 40000 --num_processes 1 --start_index 360000 

python dag_to_rdb_generator.py --output_base_dir ../RDB_datasets/RDB_large_woGNN_1hop_sub1_raw --config_file dag_to_rdb_config_large.yaml --num_rdbs 20000 --num_processes 1 --start_index 400000

python dag_to_rdb_generator.py --output_base_dir ../RDB_datasets/RDB_large_woGNN_1hop_sub2_raw --config_file dag_to_rdb_config_large.yaml --num_rdbs 20000 --num_processes 1 --start_index 480000

python dag_to_rdb_generator.py --output_base_dir ../RDB_datasets/RDB_large_wGNN_1hop_sub1_raw --use_row_gnn --config_file dag_to_rdb_config_large.yaml --num_rdbs 20000 --num_processes 1 --start_index 440000

python dag_to_rdb_generator.py --output_base_dir ../RDB_datasets/RDB_large_wGNN_1hop_sub2_raw --use_row_gnn --config_file dag_to_rdb_config_large.yaml --num_rdbs 20000 --num_processes 1 --start_index 520000