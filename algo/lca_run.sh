#!/bin/bash

set -e


LCA_DIR="GGR/code/"
IMAGES_DIR=""
ANNOTATIONS_FILE="GGR/data/annotations_LCA.json"
EMBEDDINGS_FILE="GGR/data/embeddings_LCA.pickle"
VERIFIER_PROBS="GGR/data/verifiers_probs.json"
DB_DIR="GGR/data/output"
EXP_NAME="zebra_drone"
LOG_FILE="GGR/data/output/drone_log.log"

python3 lca.py \
    "$LCA_DIR" \
    "$IMAGES_DIR" \
    "$ANNOTATIONS_FILE" \
    "$EMBEDDINGS_FILE" \
    "$VERIFIER_PROBS" \
    "$DB_DIR" \
    "$LOG_FILE" \
    "$EXP_NAME" \
    --separate_viewpoints