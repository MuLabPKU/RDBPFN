#!/usr/bin/env python3
"""
DAG to RDB Generator

This script converts DAG data into real RDBs. Given DAG structure with source/destination
nodes and table dimensions, it creates relational databases with proper table relationships.

The script focuses on child tables that have exactly 2 parent tables (excluding timestamp tables).
"""

import torch
import numpy as np
import random
import os
import time
from copy import deepcopy
from typing import Any, Dict, List, Tuple
from collections import defaultdict
import argparse
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
import yaml

from src.table_def.table_generation import (
    DataTypeConfig,
    Table,
    Relationship,
    RDB,
)


class DAGToRDBGenerator:
    """Converts DAG data into RDB structures."""

    DEFAULT_DIMENSION_CONFIG = {
        "num_rows": {
            "default": 1000,
            "parse_rules": [
                {"max_input": 100, "multiplier": 10.0, "offset": 1000},
                {"max_input": 200000, "multiplier": 0.015, "offset": 2000},
                {"multiplier": 0.0, "offset": 5000},
            ],
            "fluctuation_ratio": 0.2,
            "min": 1000,
            "max": 5000,
        },
        "num_cols": {
            "default": 5,
            "parse_rules": {
                "min_threshold": 8,
                "min_value": 12,
                "max_threshold": 8,
                "max_value": 12,
            },
            "fluctuation_ratio": 1.0,
            "min": 8,
            "max": 12,
        },
    }

    def __init__(
        self,
        dag_data_path: str,
        output_base_dir: str = "dag_generated_rdbs",
        seed: int = 42,
        use_row_gnn: bool = False,
        gnn_device: str = "cpu",
        dimension_config: Dict[str, Any] = None,
    ):
        """
        Initialize the DAG to RDB generator.

        Parameters
        ----------
        dag_data_path : str
            Path to the DAG data file (torch saved dict)
        output_base_dir : str
            Base directory for saving generated RDBs
        """
        self.dag_data_path = dag_data_path
        self.output_base_dir = output_base_dir
        self.dag_data = None
        self.parsed_dags = []
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.use_row_gnn = use_row_gnn
        self.gnn_device = gnn_device
        self.dimension_config = self._merge_dimension_config(
            deepcopy(self.DEFAULT_DIMENSION_CONFIG), dimension_config or {}
        )

    @staticmethod
    def _merge_dimension_config(
        base: Dict[str, Any], overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recursively merge user-provided dimension settings into defaults."""
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key] = DAGToRDBGenerator._merge_dimension_config(base[key], value)
            else:
                base[key] = value
        return base

    @classmethod
    def load_dimension_config(cls, config_file: str = None) -> Dict[str, Any]:
        """Load dimension settings from YAML. Returns defaults when no file is provided."""
        if config_file is None:
            return deepcopy(cls.DEFAULT_DIMENSION_CONFIG)

        with open(config_file, "r", encoding="utf-8") as f:
            loaded_config = yaml.safe_load(f) or {}

        return cls._merge_dimension_config(
            deepcopy(cls.DEFAULT_DIMENSION_CONFIG), loaded_config
        )

    def _compute_parsed_num_rows(self, raw_num_rows: int) -> int:
        row_config = self.dimension_config["num_rows"]
        for rule in row_config["parse_rules"]:
            max_input = rule.get("max_input")
            if max_input is None or raw_num_rows < max_input:
                return int(
                    raw_num_rows * rule.get("multiplier", 1.0) + rule.get("offset", 0)
                )
        return row_config["default"]

    def _compute_parsed_num_cols(self, raw_num_cols: int) -> int:
        col_config = self.dimension_config["num_cols"]
        parse_rules = col_config["parse_rules"]

        if raw_num_cols < parse_rules["min_threshold"]:
            return parse_rules["min_value"]
        if raw_num_cols > parse_rules["max_threshold"]:
            return parse_rules["max_value"]
        return int(raw_num_cols)

    @staticmethod
    def _apply_fluctuation(base_value: int, config: Dict[str, Any]) -> int:
        fluctuation_ratio = config.get("fluctuation_ratio", 0.0)
        fluctuated = int(
            base_value * (1 + random.uniform(-fluctuation_ratio, fluctuation_ratio))
        )
        return min(config["max"], max(config["min"], fluctuated))

    def load_dag_data(self):
        """Load DAG data from the saved file."""
        print(f"Loading DAG data from {self.dag_data_path}...")

        try:
            self.dag_data = torch.load(self.dag_data_path)
            print(f"Loaded DAG data with {len(self.dag_data['src_list'])} DAGs")

            # Verify data structure
            required_keys = ["src_list", "dst_list", "x_n_list", "y_list"]
            for key in required_keys:
                if key not in self.dag_data:
                    raise ValueError(f"Missing required key: {key}")

            print("DAG data structure verified successfully")

        except Exception as e:
            print(f"Error loading DAG data: {e}")
            raise

    def parse_dag_structure(self, dag_idx: int) -> Dict:
        """
        Parse a single DAG into a structured format.

        Parameters
        ----------
        dag_idx : int
            Index of the DAG to parse

        Returns
        -------
        Dict
            Parsed DAG structure with nodes, edges, and metadata
        """
        src_list = self.dag_data["src_list"][dag_idx]
        dst_list = self.dag_data["dst_list"][dag_idx]
        x_n_list = self.dag_data["x_n_list"][
            dag_idx
        ]  # [num_rows, num_cols] for each node

        # Convert to numpy arrays if they're tensors
        if torch.is_tensor(src_list):
            src_list = src_list.cpu().numpy()
        if torch.is_tensor(dst_list):
            dst_list = dst_list.cpu().numpy()
        if torch.is_tensor(x_n_list):
            x_n_list = x_n_list.cpu().numpy()

        # Get unique nodes
        all_nodes = set(src_list.tolist() + dst_list.tolist())
        num_nodes = len(all_nodes)

        # Build adjacency information
        in_degree = defaultdict(list)  # node -> list of parent nodes
        out_degree = defaultdict(list)  # node -> list of child nodes

        for src, dst in zip(src_list, dst_list):
            in_degree[dst].append(src)
            out_degree[src].append(dst)

        # Get node dimensions
        node_dimensions = {}
        for i, node in enumerate(sorted(all_nodes)):
            if i < len(x_n_list):
                node_dimensions[node] = {
                    "num_rows": self._compute_parsed_num_rows(int(x_n_list[i][0])),
                    "num_cols": self._compute_parsed_num_cols(int(x_n_list[i][1])),
                }
            else:
                # Default dimensions if not specified
                node_dimensions[node] = {
                    "num_rows": self.dimension_config["num_rows"]["default"],
                    "num_cols": self.dimension_config["num_cols"]["default"],
                }

        return {
            "dag_idx": dag_idx,
            "nodes": sorted(all_nodes),
            "edges": list(zip(src_list, dst_list)),
            "in_degree": dict(in_degree),
            "out_degree": dict(out_degree),
            "node_dimensions": node_dimensions,
            "num_nodes": num_nodes,
        }

    def find_valid_child_tables(
        self, dag_structure: Dict
    ) -> Tuple[List[int], List[int]]:
        """
        Find valid child tables and categorize them.

        Parameters
        ----------
        dag_structure : Dict
            Parsed DAG structure

        Returns
        -------
        Tuple[List[int], List[int]]
            (children_with_2_parents, children_with_other_parents)
        """
        children_with_2_parents = []
        children_with_other_parents = []

        for node in dag_structure["nodes"]:
            parents = dag_structure["in_degree"].get(node, [])
            if len(parents) == 2:
                children_with_2_parents.append(node)
            elif len(parents) > 0:  # Has at least 1 parent
                children_with_other_parents.append(node)

        return children_with_2_parents, children_with_other_parents

    def create_all_table_configs(self, dag_structure: Dict) -> List[Dict]:
        """
        Create table configurations for ALL nodes in the DAG.

        Parameters
        ----------
        dag_structure : Dict
            Parsed DAG structure

        Returns
        -------
        List[Dict]
            Table configurations for all nodes
        """
        table_configs = []

        for node in dag_structure["nodes"]:
            # Get node dimensions and parent information
            node_dims = dag_structure["node_dimensions"][node]
            # Fluctuate num_rows by 20%
            parents = dag_structure["in_degree"].get(node, [])
            num_parents = len(parents)

            # Calculate num_cols and num_features based on new logic
            num_rows = self._apply_fluctuation(
                node_dims["num_rows"], self.dimension_config["num_rows"]
            )
            original_num_cols = self._apply_fluctuation(
                node_dims["num_cols"], self.dimension_config["num_cols"]
            )
            num_features = original_num_cols  # num_features = original num_col from DAG
            num_cols = (
                original_num_cols + 1 + num_parents
            )  # num_col from DAG + 1 + parent's tables num

            # Determine if this should be a timestamp table
            # Only possible if table has exactly 2 parents, randomly determined
            is_timestamp_table = False
            if num_parents == 2:
                is_timestamp_table = random.choice([True, False])
                if is_timestamp_table:
                    num_cols += 1  # Add one more column for timestamp

            # Create table config
            table_config = {
                "name": f"table_{node}",
                "node_id": node,
                "num_rows": num_rows,
                "num_cols": num_cols,
                "num_features": num_features,
                "parent_nodes": parents,
                "num_parents": num_parents,
                "is_timestamp_table": is_timestamp_table,
            }

            table_configs.append(table_config)

        return table_configs

    def create_relationships_from_dag(
        self, dag_structure: Dict, table_configs: List[Dict]
    ) -> List[Tuple]:
        """
        Create relationships based on the full DAG graph structure.

        Parameters
        ----------
        dag_structure : Dict
            Parsed DAG structure
        table_configs : List[Dict]
            Table configurations for all nodes

        Returns
        -------
        List[Tuple]
            List of relationship tuples (from_node, from_col, to_node, to_col)
        """
        relationships = []

        # Create a mapping from node_id to table config for easy lookup
        node_to_config = {config["node_id"]: config for config in table_configs}

        # Iterate through all edges in the DAG to create relationships
        for src_node, dst_node in dag_structure["edges"]:
            # Each edge represents: parent (src) -> child (dst)
            # This means dst table has a foreign key pointing to src table's primary key

            # Find the foreign key column index in the child table
            child_config = node_to_config[dst_node]
            parent_nodes = child_config["parent_nodes"]

            # The foreign key column index depends on the order of parents
            # Column 0: primary key of child table
            # Column 1, 2, ...: foreign keys to parent tables (in order of parent_nodes)
            try:
                fk_column_index = (
                    parent_nodes.index(src_node) + 1
                )  # +1 because column 0 is PK
                relationships.append((dst_node, fk_column_index, src_node, 0))
            except ValueError:
                # This shouldn't happen if DAG structure is consistent
                print(
                    f"Warning: Parent {src_node} not found in child {dst_node} parent list"
                )
                continue

        return relationships

    def create_rdb_from_config(
        self, table_configs: List[Dict], relationships: List[Tuple], rdb_name: str
    ) -> RDB:
        """
        Create an RDB from table configurations.

        Parameters
        ----------
        table_configs : List[Dict]
            List of table configuration dictionaries
        relationships : List[Tuple]
            List of relationship tuples
        rdb_name : str
            Name for the RDB

        Returns
        -------
        RDB
            Created RDB instance
        """
        rdb = RDB(rdb_name)
        if self.use_row_gnn:
            rdb.enable_row_gnn(device=self.gnn_device)

        # Create tables
        for config in table_configs:
            # Generate column names based on new structure
            column_names = [f"{config['name']}_id"]  # Primary key (column 0)

            # Add foreign key columns
            if config["num_parents"] > 0:
                for parent_node in config["parent_nodes"]:
                    column_names.append(f"table_{parent_node}_id")

            # Add timestamp column if it's a timestamp table
            if config.get("is_timestamp_table", False):
                column_names.append("timestamp")

            # Add feature columns
            for i in range(config["num_features"]):
                column_names.append(f"feature_{i}")

            # Generate data type configs
            data_type_configs = [DataTypeConfig.primary_key_config()]  # PK (column 0)

            # Add foreign key data types
            if config["num_parents"] > 0:
                for parent_node in config["parent_nodes"]:
                    data_type_configs.append(
                        DataTypeConfig.foreign_key_config(
                            parent_table=f"table_{parent_node}"
                        )
                    )

            # Add timestamp data type if it's a timestamp table
            if config.get("is_timestamp_table", False):
                data_type_configs.append(DataTypeConfig.timestamp_config())

            # Add feature data types
            for i in range(config["num_features"]):
                if i % 2 == 0:
                    data_type_configs.append(DataTypeConfig.float_config())
                elif i % 2 == 1:
                    data_type_configs.append(
                        DataTypeConfig.categorical_config(
                            num_categories=min(
                                torch.randint(2, 10, (1,)).item(),
                                config["num_rows"],
                            ),
                        )
                    )
                else:
                    data_type_configs.append(DataTypeConfig.float_config())

            # Determine time column index if it's a timestamp table
            time_column = None
            if config.get("is_timestamp_table", False):
                # Time column is after PK and FKs but before features
                time_column = 1 + config["num_parents"]  # PK + FKs

            # Create table
            table = Table(
                num_rows=config["num_rows"],
                num_cols=config["num_cols"],
                num_features=(
                    config["num_features"] + 1
                    if config.get("is_timestamp_table", False)
                    else config["num_features"]
                ),
                column_names=column_names,
                data_type_configs=data_type_configs,
                time_column=time_column,
                device="cpu",
            )

            rdb.add_table(config["name"], table)

        # Add relationships
        for from_node, from_col, to_node, to_col in relationships:
            from_table = f"table_{from_node}"
            to_table = f"table_{to_node}"

            relationship = Relationship(from_table, from_col, to_table, to_col)
            rdb.add_relationship(relationship)

        return rdb

    @staticmethod
    def _generate_single_rdb_worker(args):
        """
        Worker function for parallel RDB generation.

        Parameters
        ----------
        args : tuple
            (rdb_index, all_dag_structures, output_base_dir, eta_min, eta_max)

        Returns
        -------
        tuple
            (success, rdb_index, result_or_error)
        """
        try:
            (
                rdb_index,
                all_dag_structures,
                output_base_dir,
                eta_min,
                eta_max,
                use_complex_tasks,
                dimension_config,
            ) = args

            # Set random seed for reproducibility (each worker gets different seed)
            random.seed(rdb_index)
            np.random.seed(rdb_index)
            torch.manual_seed(rdb_index)

            # Randomly select a DAG
            dag_idx = random.choice(range(len(all_dag_structures)))
            dag_structure = all_dag_structures[dag_idx]

            # Create generator instance (needed for access to methods)
            # We need to recreate this in the worker process
            generator = DAGToRDBGenerator(
                "",
                output_base_dir,
                seed=rdb_index,
                dimension_config=dimension_config,
            )

            # Generate configurations for ALL tables in this DAG
            table_configs = generator.create_all_table_configs(dag_structure)

            # Create relationships based on the full DAG structure
            relationships = generator.create_relationships_from_dag(
                dag_structure, table_configs
            )

            # Count timestamp tables for logging
            timestamp_tables = [
                config
                for config in table_configs
                if config.get("is_timestamp_table", False)
            ]

            # Create RDB
            rdb_name = f"dag_rdb_{rdb_index}"
            rdb = generator.create_rdb_from_config(
                table_configs, relationships, rdb_name
            )

            # Initialize SCMs with eta
            rdb.init_table_SCMs(seed=rdb_index)

            # Generate data
            rdb.generate_all_data_from_SCM()

            # Create output directory
            rdb_dir = os.path.join(output_base_dir, rdb_name)
            os.makedirs(rdb_dir, exist_ok=True)
            csv_dir = os.path.join(rdb_dir, "csv_data")
            os.makedirs(csv_dir, exist_ok=True)

            # Save to file
            rdb.save_to_file(csv_dir)

            # Initialize tasks and save to 4DBInfer format
            if use_complex_tasks:
                rdb.initialize_tasks_with_complex_tasks(
                    tasks_per_rdb=5, train_ratio=0.75, valid_ratio=0.05
                )
            else:
                rdb.initialize_tasks(
                    tasks_per_rdb=5, train_ratio=0.75, valid_ratio=0.05
                )

            # Save to 4DBInfer format with tasks
            rdb.save_to_4dbinfer_dataset_with_tasks(rdb_dir)

            # Return success info
            return (
                True,
                rdb_index,
                {
                    "rdb_name": rdb_name,
                    "dag_idx": dag_structure["dag_idx"],
                    "num_tables": len(table_configs),
                    "num_timestamp_tables": len(timestamp_tables),
                    "num_relationships": len(relationships),
                    "rdb_dir": rdb_dir,
                },
            )

        except Exception as e:
            import traceback

            return (False, rdb_index, str(e) + "\n" + traceback.format_exc())

    def generate_rdbs_from_dags(
        self,
        num_rdbs: int = 1,
        eta_min: float = 0.1,
        eta_max: float = 10.0,
        start_index: int = 0,
        num_processes: int = None,
        use_complex_tasks: bool = False,
    ) -> List[RDB]:
        """
        Generate RDBs from the loaded DAG data.

        Parameters
        ----------
        num_rdbs : int
            Number of RDBs to generate
        eta_min : float
            Minimum eta value for enhanced temporal sampling
        eta_max : float
            Maximum eta value for enhanced temporal sampling
        start_index : int
            Start index for the RDBs to generate
        num_processes : int, optional
            Number of parallel processes to use. If None, uses all available CPU cores.
            If 1, runs sequentially (original behavior).
        use_complex_tasks : bool
            Whether to use complex tasks
        Returns
        -------
        List[RDB]
            List of generated RDBs
        """
        if self.dag_data is None:
            self.load_dag_data()

        print(f"Generating {num_rdbs} RDBs from DAG data...")
        os.makedirs(self.output_base_dir, exist_ok=True)

        # Parse all DAGs
        print("Parsing DAG structures...")
        all_dag_structures = []
        for i in range(len(self.dag_data["src_list"])):
            try:
                dag_structure = self.parse_dag_structure(i)
                all_dag_structures.append(dag_structure)
            except Exception as e:
                print(f"Error parsing DAG {i}: {e}")
                continue

        print(f"Successfully parsed {len(all_dag_structures)} DAGs")

        if len(all_dag_structures) == 0:
            print("No valid DAGs found!")
            return []

        # Determine number of processes
        if num_processes is None:
            num_processes = cpu_count()
        elif num_processes <= 0:
            num_processes = 1

        print(f"Using {num_processes} processes for parallel generation")

        # Prepare arguments for parallel processing
        worker_args = [
            (
                i,
                all_dag_structures,
                self.output_base_dir,
                eta_min,
                eta_max,
                use_complex_tasks,
                self.dimension_config,
            )
            for i in range(start_index, start_index + num_rdbs)
        ]

        generated_rdbs = []
        successful_generations = 0

        if num_processes == 1:
            # Sequential processing (original behavior)
            print("Running in sequential mode...")
            for args in worker_args:
                success, rdb_index, result = self._generate_single_rdb_worker(args)
                if success:
                    info = result
                    print(
                        f"Generating RDB {rdb_index + 1}/{num_rdbs}: {info['rdb_name']} "
                        f"(DAG {info['dag_idx']}, {info['num_tables']} tables, "
                        f"{info['num_timestamp_tables']} timestamp tables, "
                        f"{info['num_relationships']} relationships)"
                    )
                    print(f"  ✓ Saved to {info['rdb_dir']}")
                    successful_generations += 1
                    # Note: We don't append the actual RDB object in parallel mode to save memory
                else:
                    print(f"  ✗ Error generating RDB {rdb_index}: {result}")
        else:
            # Parallel processing
            print("Running in parallel mode...")
            print("Note: RDB objects are not returned in parallel mode to save memory")

            try:
                with Pool(processes=num_processes) as pool:
                    # Use map to process all arguments
                    results = pool.map(self._generate_single_rdb_worker, worker_args)

                    # Process results
                    for success, rdb_index, result in results:
                        if success:
                            info = result
                            print(
                                f"✓ RDB {rdb_index + 1}/{num_rdbs}: {info['rdb_name']} "
                                f"(DAG {info['dag_idx']}, {info['num_tables']} tables, "
                                f"{info['num_timestamp_tables']} timestamp tables, "
                                f"{info['num_relationships']} relationships) -> {info['rdb_dir']}"
                            )
                            successful_generations += 1
                        else:
                            print(f"✗ Error generating RDB {rdb_index + 1}: {result}")
            except Exception as e:
                print(f"Error in parallel processing: {e}")
                print("Falling back to sequential processing...")
                # Fallback to sequential processing
                for args in worker_args:
                    success, rdb_index, result = self._generate_single_rdb_worker(args)
                    if success:
                        info = result
                        print(
                            f"Generating RDB {rdb_index + 1}/{num_rdbs}: {info['rdb_name']} "
                            f"(DAG {info['dag_idx']}, {info['num_tables']} tables, "
                            f"{info['num_timestamp_tables']} timestamp tables, "
                            f"{info['num_relationships']} relationships)"
                        )
                        print(f"  ✓ Saved to {info['rdb_dir']}")
                        successful_generations += 1
                    else:
                        print(f"  ✗ Error generating RDB {rdb_index}: {result}")

        print(
            f"\nGeneration complete: {successful_generations}/{num_rdbs} RDBs generated successfully"
        )
        return generated_rdbs

    def analyze_dag_statistics(self):
        """Analyze statistics of the loaded DAG data."""
        if self.dag_data is None:
            self.load_dag_data()

        print("\nDAG DATA STATISTICS")
        print("=" * 50)

        num_dags = len(self.dag_data["src_list"])
        print(f"Total DAGs: {num_dags}")

        # Analyze each DAG
        node_counts = []
        edge_counts = []
        valid_child_counts = []
        timestamp_table_counts = []
        total_table_counts = []

        for i in range(num_dags):
            try:
                dag_structure = self.parse_dag_structure(i)

                # Generate table configs to see how many tables we'll create
                table_configs = self.create_all_table_configs(dag_structure)
                timestamp_tables = [
                    config
                    for config in table_configs
                    if config.get("is_timestamp_table", False)
                ]

                # Analyze traditional valid children (for comparison)
                children_with_2_parents, children_with_other_parents = (
                    self.find_valid_child_tables(dag_structure)
                )
                total_valid_children = len(children_with_2_parents) + len(
                    children_with_other_parents
                )

                node_counts.append(dag_structure["num_nodes"])
                edge_counts.append(len(dag_structure["edges"]))
                valid_child_counts.append(total_valid_children)
                timestamp_table_counts.append(len(timestamp_tables))
                total_table_counts.append(len(table_configs))

            except Exception as e:
                print(f"Error analyzing DAG {i}: {e}")
                continue

        if node_counts:
            print(
                f"Nodes per DAG: min={min(node_counts)}, max={max(node_counts)}, avg={np.mean(node_counts):.1f}"
            )
            print(
                f"Edges per DAG: min={min(edge_counts)}, max={max(edge_counts)}, avg={np.mean(edge_counts):.1f}"
            )
            print(
                f"Tables per RDB: min={min(total_table_counts)}, max={max(total_table_counts)}, avg={np.mean(total_table_counts):.1f}"
            )
            print(
                f"Timestamp tables per RDB: min={min(timestamp_table_counts)}, max={max(timestamp_table_counts)}, avg={np.mean(timestamp_table_counts):.1f}"
            )
            print(
                f"Valid children per DAG (old method): min={min(valid_child_counts)}, max={max(valid_child_counts)}, avg={np.mean(valid_child_counts):.1f}"
            )
            print(f"Total possible tables across all DAGs: {sum(total_table_counts)}")
            print(f"Total possible timestamp tables: {sum(timestamp_table_counts)}")

            # Show distribution of parent counts
            parent_count_distribution = {}
            tables_with_2_parents = 0

            for i in range(num_dags):
                try:
                    dag_structure = self.parse_dag_structure(i)
                    for node in dag_structure["nodes"]:
                        num_parents = len(dag_structure["in_degree"].get(node, []))
                        parent_count_distribution[num_parents] = (
                            parent_count_distribution.get(num_parents, 0) + 1
                        )
                        if num_parents == 2:
                            tables_with_2_parents += 1
                except Exception:
                    continue

            print("\nParent count distribution:")
            for parent_count in sorted(parent_count_distribution.keys()):
                count = parent_count_distribution[parent_count]
                print(f"  {parent_count} parents: {count} tables")

            print(
                f"\nTables eligible for timestamp (2 parents): {tables_with_2_parents}"
            )
            print(
                f"Expected timestamp tables (50% random): ~{tables_with_2_parents // 2}"
            )


def main():
    """Main function to generate RDBs from DAG data."""
    # Configuration
    print("=" * 40)
    print("DAG TO RDB GENERATOR")
    print("=" * 40)

    # Check if DAG data file exists
    if not os.path.exists(dag_data_path):
        print(f"Error: DAG data file not found at {dag_data_path}")
        print(
            "Please update the dag_data_path variable with the correct path to your DAG data file."
        )
        return

    # Initialize generator
    generator = DAGToRDBGenerator(
        dag_data_path,
        output_base_dir,
        seed=random_seed,
        use_row_gnn=use_row_gnn,
        gnn_device=gnn_device,
        dimension_config=dimension_config,
    )

    # Analyze DAG statistics
    try:
        generator.analyze_dag_statistics()
    except Exception as e:
        print(f"Error analyzing DAG statistics: {e}")

    # Generate RDBs
    try:
        start_time = time.time()
        _ = generator.generate_rdbs_from_dags(
            num_rdbs=num_rdbs_to_generate,
            eta_min=eta_min,
            eta_max=eta_max,
            num_processes=num_processes,
            start_index=start_index,
            use_complex_tasks=use_complex_tasks,
        )
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Results saved in: {generator.output_base_dir}")
        print(
            f"\nTotal execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)"
        )

    except Exception as e:
        print(f"Error generating RDBs: {e}")


if __name__ == "__main__":
    # Set multiprocessing start method for cross-platform compatibility
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Generate RDBs from DAG data with optional parallel processing"
    )
    parser.add_argument(
        "--num_rdbs", type=int, default=1000, help="Number of RDBs to generate"
    )
    parser.add_argument(
        "--eta_min",
        type=float,
        default=0.1,
        help="Minimum eta value for temporal sampling",
    )
    parser.add_argument(
        "--eta_max",
        type=float,
        default=10.0,
        help="Maximum eta value for temporal sampling",
    )
    parser.add_argument(
        "--dag_data_path",
        type=str,
        default="datasets/rdb_v1.pth",
        help="Path to DAG data file",
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default="dag_generated_rdbs",
        help="Output directory for generated RDBs",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="dag_to_rdb_config_small.yaml",
        help="YAML config file for num_rows/num_cols sizing rules",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of parallel processes (default: 1 for sequential)",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start index for the RDBs to generate",
    )
    parser.add_argument(
        "--use_complex_tasks",
        type=bool,
        default=False,
        help="Whether to use complex tasks",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--use_row_gnn",
        action="store_true",
        help="Enable row-level GNN refinement before column conversion",
    )
    parser.add_argument(
        "--gnn_device",
        type=str,
        default="cpu",
        help="Device for row-level GNN (e.g., 'cpu', 'cuda:0')",
    )
    args = parser.parse_args()

    num_rdbs_to_generate = args.num_rdbs
    eta_min = args.eta_min
    eta_max = args.eta_max
    dag_data_path = args.dag_data_path
    output_base_dir = args.output_base_dir
    config_file = args.config_file
    num_processes = args.num_processes
    start_index = args.start_index
    use_complex_tasks = args.use_complex_tasks
    use_row_gnn = args.use_row_gnn
    random_seed = args.random_seed
    gnn_device = args.gnn_device

    # Validate num_processes
    if num_processes is not None and num_processes <= 0:
        print(
            "Warning: num_processes must be positive. Setting to 1 (sequential mode)."
        )
        num_processes = 1

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    dimension_config = DAGToRDBGenerator.load_dimension_config(config_file)

    main()
