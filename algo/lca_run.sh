# USE: Run LCA separate from the pipeline. RUN FROM GGR

#!/bin/bash

set -e

ALG_NAME = "hdbscan"
IMAGES_DIR=""
ANNOTATIONS_FILE="test_dataset/annotations_LCA.json"
EMBEDDINGS_FILE="test_dataset/embeddings_LCA.pickle"
VERIFIER_PROBS="test_dataset/verifiers_probs.json"
DB_DIR="test_dataset/lca/db"
EXP_NAME="zebra_drone"
LOG_FILE="test_dataset/lca/drone_log.log"

python3 algo/lca.py \
    "$LCA_DIR" \
    "$IMAGES_DIR" \
    "$ANNOTATIONS_FILE" \
    "$EMBEDDINGS_FILE" \
    "$VERIFIER_PROBS" \
    "$DB_DIR" \
    "$LOG_FILE" \
    "$EXP_NAME" \
    "$ALG_NAME" \
    --separate_viewpoints