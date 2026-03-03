#!/bin/bash

DATASET_DIR=data/pascal

# Download PASCAL VOC12 (2GB)#!/bin/bash
curl -L -o ./pascal-voc-2012-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/gopalbhattrai/pascal-voc-2012-dataset

mkdir -p $DATASET_DIR

unzip ./pascal-voc-2012-dataset.zip -d $DATASET_DIR
rm ./pascal-voc-2012-dataset.zip

rm -r $DATASET_DIR/VOC2012_test/
mv $DATASET_DIR/VOC2012_train_val/VOC2012_train_val/* $DATASET_DIR
rm -r $DATASET_DIR/VOC2012_train_val

echo "PASCAL VOC12 download and extraction complete."

# Download augmented segmentation masks

echo "Downloading augmented segmentation masks..."
gdown 1tci2rpNj9S1FqTPvew4FOrkBVzbU72dr
unzip SegmentationClassAug.zip -d $DATASET_DIR
rm SegmentationClassAug.zip

echo "PASCAL VOC12 augmented segmentation masks download and extraction complete."