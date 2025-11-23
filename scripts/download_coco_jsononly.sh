mkdir -p data/coco

wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip -d data/coco
rm annotations_trainval2014.zip

echo "COCO JSON annotations download and setup complete."

echo "Renaming COCO category IDs in annotation files..."
python affex/data/preprocessing.py --instances_path data/coco/annotations/instances_train2014.json
python affex/data/preprocessing.py --instances_path data/coco/annotations/instances_val2014.json