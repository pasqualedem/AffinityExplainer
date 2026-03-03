#!/usr/bin/env bash
set -e

echo "Navigating to data directory..."
cd data && mkdir -p coco && cd coco

echo "Downloading COCO 2017 train images..."
wget http://images.cocodataset.org/zips/train2017.zip

echo "Downloading COCO 2017 val images..."
wget http://images.cocodataset.org/zips/val2017.zip

echo "Downloading COCO 2014 annotations..."
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip

echo "Extracting zip files..."
unzip "*.zip" && echo "Extraction finished."

echo "Removing zip archives..."
rm *.zip

echo "Organizing directories..."
mv val2017/* train2017/ && mv train2017 train_val_2017 && rm -rf val2017

echo "COCO download and setup complete."

cd ../../..

echo "Renaming COCO category IDs in annotation files..."

python affex/data/preprocessing.py --instances_path data/coco/annotations/instances_train2014.json
python affex/data/preprocessing.py --instances_path data/coco/annotations/instances_val2014.json