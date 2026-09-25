import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `affex`
r"""Train the PANet-style prototype head on the frozen DCAMA (Swin) backbone.

Same training regime as DCAMA (frozen encoder, head-only, episodic 1-shot PASCAL-5i),
using the original DCAMA data pipeline. Quick-and-early-stopped: the goal is a working
model for the reviewer M83E same-encoder/different-head comparison, not SOTA.

Usage:
  python train_panet.py --datapath <pascal_root_with_JPEGImages_and_SegmentationClassAug> \
      --fold 0 --bsz 16 --lr 1e-3 --nepoch 30 --patience 3
"""
import argparse
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from affex.models.panet_head import PANetMultiClass
from affex.utils.utils import ResultDict
from train_panet_data.pascal import DatasetPASCAL


def build_loaders(datapath, fold, bsz, nworker):
    tfm = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    trn = DatasetPASCAL(datapath, fold=fold, transform=tfm, split='trn', shot=1,
                        use_original_imgsize=False)
    val = DatasetPASCAL(datapath, fold=fold, transform=tfm, split='val', shot=1,
                        use_original_imgsize=False)
    trn_loader = DataLoader(trn, batch_size=bsz, shuffle=True, num_workers=nworker,
                            drop_last=True)
    val_loader = DataLoader(val, batch_size=bsz, shuffle=False, num_workers=nworker)
    return trn_loader, val_loader


def run_epoch(model, loader, optimizer, device, training, max_batches=None):
    model.model.train(training)
    model.feature_extractor.eval()
    losses = []
    inter = {}
    union = {}
    for idx, batch in enumerate(loader):
        if max_batches is not None and idx >= max_batches:
            break
        query_img = batch['query_img'].to(device)
        query_mask = batch['query_mask'].to(device)
        support_img = batch['support_imgs'].squeeze(1).to(device)
        support_mask = batch['support_masks'].squeeze(1).to(device)

        result = model.forward_episode(query_img, support_img, support_mask,
                                       query_mask=query_mask)
        loss = result[ResultDict.LOSS]
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(loss.item())

        logits = result[ResultDict.LOGITS]
        logits = torch.nn.functional.interpolate(
            logits, query_mask.shape[-2:], mode='bilinear', align_corners=True)
        pred = logits.argmax(dim=1)
        for b in range(pred.shape[0]):
            cid = int(batch['class_id'][b])
            p = pred[b].bool()
            g = query_mask[b].bool()
            inter[cid] = inter.get(cid, 0) + (p & g).sum().item()
            union[cid] = union.get(cid, 0) + (p | g).sum().item()
        if idx % 50 == 0:
            print(f"  [{'trn' if training else 'val'}] batch {idx}/{len(loader)} "
                  f"loss {np.mean(losses):.4f}", flush=True)

    ious = [inter[c] / union[c] for c in inter if union[c] > 0]
    miou = 100.0 * float(np.mean(ious)) if ious else 0.0
    return float(np.mean(losses)), miou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, required=True)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--bsz', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--nepoch', type=int, default=30)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--nworker', type=int, default=8)
    parser.add_argument('--backbone_checkpoint', type=str,
                        default='checkpoints/dcama/swin_base_patch4_window12_384.pth')
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--max_batches', type=int, default=None,
                        help='cap batches per epoch (smoke tests)')
    parser.add_argument('--val_batches', type=int, default=None,
                        help='cap validation batches per epoch')
    args = parser.parse_args()

    out = args.out or f'checkpoints/panet/pascal/swin_fold{args.fold}.pt'
    os.makedirs(os.path.dirname(out), exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PANetMultiClass('swin', args.backbone_checkpoint, image_size=384)
    model.to(device)

    optimizer = optim.SGD(model.get_learnable_params(), lr=args.lr, momentum=0.9,
                          weight_decay=args.lr / 10, nesterov=True)
    trn_loader, val_loader = build_loaders(args.datapath, args.fold, args.bsz, args.nworker)

    best_miou = float('-inf')
    bad_epochs = 0
    for epoch in range(args.nepoch):
        np.random.seed()  # episodic sampling randomness for training
        trn_loss, trn_miou = run_epoch(model, trn_loader, optimizer, device,
                                       training=True, max_batches=args.max_batches)
        np.random.seed(0)  # frozen val episodes
        with torch.no_grad():
            val_loss, val_miou = run_epoch(model, val_loader, optimizer, device,
                                           training=False, max_batches=args.val_batches)
        print(f'epoch {epoch}: trn loss {trn_loss:.4f} miou {trn_miou:.2f} | '
              f'val loss {val_loss:.4f} miou {val_miou:.2f} '
              f'(temp {model.model.temperature.item():.2f})', flush=True)

        if val_miou > best_miou:
            best_miou = val_miou
            bad_epochs = 0
            torch.save(model.model.state_dict(), out)
            print(f'  saved best head -> {out} (val miou {val_miou:.2f})', flush=True)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f'early stop at epoch {epoch}; best val miou {best_miou:.2f}', flush=True)
                break

    print(f'done; best val miou {best_miou:.2f}; checkpoint {out}', flush=True)


if __name__ == '__main__':
    main()
