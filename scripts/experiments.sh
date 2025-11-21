# COCO
python main.py grid --parameters parameters/coco/iauc_dauc_dcama.yaml --parallel
python main.py grid --parameters parameters/coco/iauc_dauc_dmtnet.yaml --parallel
python main.py grid --parameters parameters/coco/cut_iauc.yaml --parallel
python main.py grid --parameters parameters/coco/cut_iauc_miou.yaml --parallel
python main.py grid --parameters parameters/coco/cut_iauc_miou_N1K1.yaml --parallel

# PASCAL
python main.py grid --parameters parameters/pascal/iauc_dauc_dcama.yaml --parallel
python main.py grid --parameters parameters/pascal/iauc_dauc_dmtnet.yaml --parallel
python main.py grid --parameters parameters/pascal/cut_iauc.yaml --parallel
python main.py grid --parameters parameters/pascal/cut_iauc_miou.yaml --parallel
python main.py grid --parameters parameters/pascal/cut_iauc_miou_N1K1.yaml --parallel

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
python main.py grid --parameters parameters/cross/cut_iauc_miou_deepglobe_N1K1.yaml --parallel
python main.py grid --parameters parameters/cross/cut_iauc_miou_lung_N1K1.yaml --parallel
python main.py grid --parameters parameters/cross/cut_iauc_miou_isic_N1K1.yaml --parallel

# Feature Ablation
python main.py grid --parameters parameters/feature_ablation.yaml --parallel