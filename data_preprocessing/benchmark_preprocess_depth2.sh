#!/bin/bash

ROOT_DIR=$1
MAX_JOBS=${2:-4}  # Default to 4 parallel jobs if not specified

if [ -z "$ROOT_DIR" ]; then
    echo "Usage: $0 ROOT_DIR [MAX_JOBS]"
    echo "  ROOT_DIR: Root directory containing datasets"
    echo "  MAX_JOBS: Number of parallel jobs (default: 4)"
    exit 1
fi

if [ ! -d "$ROOT_DIR" ]; then
    echo "Error: ROOT_DIR '$ROOT_DIR' does not exist"
    exit 1
fi

echo "Running depth=2 preprocessing on datasets in parallel (max $MAX_JOBS jobs)"

# Function to run depth=2 preprocessing for a dataset
process_dataset_depth2() {
    local root_dir=$1
    local dataset=$2
    
    echo "Starting dataset: $dataset (depth=2)"
    
    # Run depth 2 (DFS processing) only
    bash preprocess_script.sh "$root_dir" "$dataset" 2
    
    echo "Completed dataset: $dataset (depth=2)"
}

# Export function so it's available to parallel processes
export -f process_dataset_depth2

# Create a list of datasets to process
dataset_list=$(mktemp)
for dataset in $ROOT_DIR/*; do
    if [ -d "$dataset" ]; then
        dataset=$(basename "$dataset")
        echo "$ROOT_DIR $dataset" >> "$dataset_list"
    fi
done

# Count datasets
num_datasets=$(wc -l < "$dataset_list")
echo "Found $num_datasets datasets to process with depth=2"

# Run datasets in parallel
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for dataset processing"
    parallel -j "$MAX_JOBS" --colsep ' ' process_dataset_depth2 {1} {2} :::: "$dataset_list"
else
    echo "GNU parallel not found, using xargs with background processes"
    xargs -n 2 -P "$MAX_JOBS" -I {} bash -c 'process_dataset_depth2 {}' < "$dataset_list"
fi

# Clean up
rm "$dataset_list"

echo "All datasets processed successfully with depth=2"
