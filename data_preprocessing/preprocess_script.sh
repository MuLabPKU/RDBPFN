#!/bin/bash

ROOT_DIR=$1
DATASET=$2
DEPTH=$3
if [ -z "$DEPTH" ]; then
    DEPTH=0
fi

mkdir -p $ROOT_DIR-tmp
mkdir -p $ROOT_DIR-processed


if [ $DEPTH -eq 0 ]; then
    # First check if the dataset is already preprocessed
    if [ -d $ROOT_DIR-processed/$DATASET-single ]; then
        echo "Dataset $DATASET-single already preprocessed"
        exit 0
    fi
    python -m tab2graph.main preprocess $ROOT_DIR/$DATASET transform $ROOT_DIR-processed/$DATASET-single -c configs/transform/single.yaml
else
    # First check if the dataset is already preprocessed
    if [ -d $ROOT_DIR-processed/$DATASET-dfs-$DEPTH ]; then
        echo "Dataset $DATASET-dfs-$DEPTH already preprocessed"
        exit 0
    fi
    python -m tab2graph.main preprocess $ROOT_DIR/$DATASET transform $ROOT_DIR-tmp/$DATASET-pre-dfs -c configs/transform/pre-dfs.yaml
    # Run Deep Feature Synthesis
    python -m tab2graph.main preprocess $ROOT_DIR-tmp/$DATASET-pre-dfs dfs $ROOT_DIR-tmp/$DATASET-post-dfs -c configs/dfs/dfs-$DEPTH-sql.yaml
    # Post-dfs processing including data normalization, extra featurization, etc.
    python -m tab2graph.main preprocess $ROOT_DIR-tmp/$DATASET-post-dfs transform $ROOT_DIR-processed/$DATASET-dfs-$DEPTH -c configs/transform/post-dfs.yaml
fi