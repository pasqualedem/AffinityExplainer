# COCO
python main.py grid --parameters parameters/coco/iauc_dauc_miou_N1K5.yaml
python main.py grid --parameters parameters/coco/cut_iauc_N1K1.yaml
python main.py grid --parameters parameters/coco/cut_iauc_N1K5.yaml
python main.py grid --parameters parameters/coco/cut_iauc_miou_N1K5.yaml
python main.py grid --parameters parameters/coco/cut_iauc_miou_N1K1.yaml

# PASCAL
python main.py grid --parameters parameters/pascal/iauc_dauc_miou_N1K5.yaml
python main.py grid --parameters parameters/pascal/cut_iauc_N1K1.yaml
python main.py grid --parameters parameters/pascal/cut_iauc_N1K5.yaml
python main.py grid --parameters parameters/pascal/cut_iauc_miou_N1K5.yaml
python main.py grid --parameters parameters/pascal/cut_iauc_miou_N1K1.yaml

# Ablation
python main.py grid --parameters parameters/feature_ablation.yaml
python main.py grid --parameters parameters/blur_ablation.yaml
python main.py grid --parameters parameters/ablation/dcama_ablation.yaml
python main.py grid --parameters parameters/ablation/iauc_size.yaml