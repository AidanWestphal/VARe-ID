# USE: Runs post processing using the old data format. MANUALLY ADD YOUR FIELDS HERE
# RUN FROM GGR

set -e

COMMAND="old"
IMAGES_DIR="test_dataset/images"
EMBEDDINGS_FILE="test_dataset/embeddings_LCA.pickle"
TIMESTAMP_FILE="test_dataset/merged_detections_with_timestamps_dji_0100_0101_0102.csv"
METADATA_FILE="test_dataset/ca_filtered_merged_tracking_ids_dji_0100_0101_0102_no_unwanted_viewpoints_pre_miewid_with_ids.csv"
MERGED_ANNOTS_FILE="test_dataset/output/merged_bbox.csv"
LEFT_LCA_FILE="test_dataset/lca/db/annotations_LCA_LCA_left.json"
RIGHT_LCA_FILE="test_dataset/lca/db/annotations_LCA_LCA_right.json"
LEFT_LCA_OUT="test_dataset/output/LCA_left.json"
RIGHT_LCA_OUT="test_dataset/output/LCA_right.json"

python3 algo/LCA_postprocessing_evaluation.py \
    "$COMMAND" \
    "$IMAGES_DIR" \
    "$EMBEDDINGS_FILE" \
    "$TIMESTAMP_FILE" \
    "$METADATA_FILE" \
    "$MERGED_ANNOTS_FILE" \
    "$LEFT_LCA_FILE" \
    "$RIGHT_LCA_FILE" \
    "$LEFT_LCA_OUT" \
    "$RIGHT_LCA_OUT"
