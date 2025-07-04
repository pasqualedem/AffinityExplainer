# Data Generation Script

python main.py generate -p parameters/data/pascal.yaml
python main.py generate -p parameters/data/coco.yaml

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