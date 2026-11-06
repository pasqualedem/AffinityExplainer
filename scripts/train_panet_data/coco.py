r""" COCO-20i few-shot semantic segmentation dataset, read through pycocotools.

Same interface and conventions as DCAMA's (HSNet's) DatasetCOCO, so that the prototype
head is trained on COCO the way DCAMA's COCO checkpoints were:
  * 80 classes = COCO categories sorted by id; fold i holds out the interleaved classes
    {i, i+4, i+8, ...} (class_ids_val), and trains on the other 60;
  * training episodes come from train2014, validation episodes from val2014;
  * an episode draws a class at random, then the query and the supports among the images
    of that class. The draw uses a generator seeded with (self.seed, index): DataLoader
    workers would otherwise share one NumPy state and repeat each other's episodes. The
    trainer sets a new seed for every training epoch; validation keeps seed 0, which
    freezes its 1000 episodes;
  * an (image, class) pair counts only if the class covers at least 2*32*32 pixels,
    the filter of the PFENet lists HSNet uses.
Images are read from the COCO 2017 folder by image id (2017 train+val = 2014 train+val).
"""
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import PIL.Image as Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from torch.utils.data import Dataset

MIN_AREA = 2 * 32 * 32


class DatasetCOCO(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize=False,
                 img_dir='train_val_2017'):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 80
        self.benchmark = 'coco'
        self.shot = shot
        self.transform = transform
        self.use_original_imgsize = use_original_imgsize
        self.img_path = os.path.join(datapath, img_dir)
        self.seed = 0

        year_split = 'train2014' if self.split == 'trn' else 'val2014'
        coco = COCO(os.path.join(datapath, 'annotations', f'instances_{year_split}.json'))
        cat_ids = sorted(coco.getCatIds())
        self.class_ids = self.build_class_ids()
        keep = {cat_ids[c]: c for c in self.class_ids}

        # (image id, class) -> polygons/RLEs of its non-crowd instances
        self.segs = defaultdict(list)
        area = defaultdict(float)
        for ann in coco.dataset['annotations']:
            c = keep.get(ann['category_id'])
            if c is None or ann.get('iscrowd', 0):
                continue
            self.segs[(ann['image_id'], c)].append(ann['segmentation'])
            area[(ann['image_id'], c)] += ann['area']
        self.img_size = {i['id']: (i['height'], i['width']) for i in coco.dataset['images']}
        self.img_metadata = sorted(k for k, a in area.items() if a >= MIN_AREA)
        self.img_metadata_classwise = defaultdict(list)
        for img_id, c in self.img_metadata:
            self.img_metadata_classwise[c].append(img_id)
        del coco
        print(f'COCO-20i fold {fold} {self.split}: {len(self.class_ids)} classes, '
              f'{len(self.img_metadata)} (image, class) pairs', flush=True)

    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else 1000

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold + self.nfolds * v for v in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]
        return class_ids_trn if self.split == 'trn' else class_ids_val

    def sample_episode(self, idx):
        rng = np.random.default_rng([self.seed, idx])
        class_sample = int(rng.choice(self.class_ids))
        pool = self.img_metadata_classwise[class_sample]
        query_id = int(rng.choice(pool))
        support_ids = []
        while len(support_ids) < self.shot:
            s = int(rng.choice(pool))
            if s != query_id and s not in support_ids:
                support_ids.append(s)
        return query_id, support_ids, class_sample

    def read_img(self, img_id):
        return Image.open(os.path.join(self.img_path, f'{img_id:012d}.jpg')).convert('RGB')

    def read_mask(self, img_id, class_id):
        h, w = self.img_size[img_id]
        m = np.zeros((h, w), dtype=np.uint8)
        for seg in self.segs[(img_id, class_id)]:
            rle = mask_utils.frPyObjects(seg, h, w) if isinstance(seg, list) else seg
            if isinstance(rle, list):
                rle = mask_utils.merge(rle)
            elif isinstance(rle.get('counts'), list):
                rle = mask_utils.frPyObjects(rle, h, w)
            m |= mask_utils.decode(rle).astype(np.uint8)
        return torch.from_numpy(m)

    def __getitem__(self, idx):
        query_id, support_ids, class_sample = self.sample_episode(idx)
        query_img = self.read_img(query_id)
        org_qry_imsize = query_img.size
        query_img = self.transform(query_img)
        query_mask = self.read_mask(query_id, class_sample).float()
        if not self.use_original_imgsize:
            query_mask = F.interpolate(query_mask[None, None], query_img.shape[-2:],
                                       mode='nearest').squeeze()

        support_imgs = torch.stack([self.transform(self.read_img(s)) for s in support_ids])
        support_masks = torch.stack([
            F.interpolate(self.read_mask(s, class_sample).float()[None, None],
                          support_imgs.shape[-2:], mode='nearest').squeeze()
            for s in support_ids])

        return {'query_img': query_img,
                'query_mask': query_mask,
                'query_name': str(query_id),
                'query_ignore_idx': torch.zeros_like(query_mask),
                'org_query_imsize': org_qry_imsize,
                'support_imgs': support_imgs,
                'support_masks': support_masks,
                'support_names': [str(s) for s in support_ids],
                'support_ignore_idxs': torch.zeros_like(support_masks),
                'class_id': torch.tensor(class_sample)}
