# Data Generation Script

python main.py generate -p parameters/data/pascal.yaml
python main.py generate -p parameters/data/coco.yaml
python main.py generate -p parameters/data/isic.yaml
python main.py generate -p parameters/data/deepglobe.yaml
python main.py generate -p parameters/data/lung.yaml
python main.py generate -p parameters/data/pascal_N1K1.yaml
python main.py generate -p parameters/data/coco_N1K1.yaml
python main.py generate -p parameters/data/isic_N1K1.yaml
python main.py generate -p parameters/data/deepglobe_N1K1.yaml
python main.py generate -p parameters/data/lung_N1K1.yaml

# COCO
python main.py grid --parameters parameters/coco/iauc_dauc_dcama.yaml --parallel
python main.py grid --parameters parameters/coco/iauc_dauc_dmtnet.yaml --parallel
python main.py grid --parameters parameters/coco/cut_iauc.yaml --parallel
python main.py grid --parameters parameters/coco/cut_iauc_miou.yaml --parallel

# PASCAL
python main.py grid --parameters parameters/pascal/iauc_dauc_dcama.yaml --parallel
python main.py grid --parameters parameters/pascal/iauc_dauc_dmtnet.yaml --parallel
python main.py grid --parameters parameters/pascal/cut_iauc.yaml --parallel
python main.py grid --parameters parameters/pascal/cut_iauc_miou.yaml --parallel

# Cross Domain
python main.py grid --parameters parameters/cross/iauc_dauc_deepglobe.yaml --parallel
python main.py grid --parameters parameters/cross/iauc_dauc_lung.yaml --parallel
python main.py grid --parameters parameters/cross/iauc_dauc_isic.yaml --parallel

python main.py grid --parameters parameters/cross/cut_iauc_deepglobe.yaml --parallel
python main.py grid --parameters parameters/cross/cut_iauc_lung.yaml --parallel
python main.py grid --parameters parameters/cross/cut_iauc_isic.yaml --parallel

python main.py grid --parameters parameters/cross/cut_iauc_miou_deepglobe.yaml --parallel
python main.py grid --parameters parameters/cross/cut_iauc_miou_lung.yaml --parallel
python main.py grid --parameters parameters/cross/cut_iauc_miou_isic.yaml --parallel