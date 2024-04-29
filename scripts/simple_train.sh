# Copyright (C) 2024 Denso IT Laboratory, Inc.
# All Rights Reserved

COLMAP_RESULTS_DIR=$3
DATASET_ROOT=$4
IMAGE_LIST_DIR=$5
OUTPUT_DIR=$6

for i in `seq -f '%05g' $1 $2`; do
    python gaussian-splatting/train.py -s $COLMAP_RESULTS_DIR/$i -i $DATASET_ROOT/train/rgbs -w -m $OUTPUT_DIR/$i
done
