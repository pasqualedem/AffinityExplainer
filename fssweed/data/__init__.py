import torch

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from fssweed.data.dataset import FSSDataset, VariableBatchSampler
from fssweed.data.deepglobe import DatasetDeepglobe
from fssweed.data.evican import EVICAN
from fssweed.data.industrial import DatasetIndustrial
from fssweed.data.isic import DatasetISIC
from fssweed.data.kvasir import KvasirTestDataset
from fssweed.data.lab2wild import Lab2Wild
from fssweed.data.lung import LungCancer
from fssweed.data.nucleus import Nucleus
from fssweed.data.test_pascal import TestDatasetPASCAL
from fssweed.data.pothole import Pothole
from fssweed.data.transforms import Normalize, Resize

from fssweed.data.coco import CocoLVISDataset
from fssweed.data.coco_crop import CocoLVISCrop
from fssweed.data.utils import get_mean_std

from fssweed.data.weedmap import WeedMapTestDataset


TEST_DATASETS = {
    "test_weedmap": WeedMapTestDataset,
    "test_deepglobe": DatasetDeepglobe,
    "test_isic": DatasetISIC,
    "test_evican": EVICAN,
    "test_nucleus": Nucleus,
    "test_pothole": Pothole,
    "test_lab2wild": Lab2Wild,
    "test_lungcancer": LungCancer,
    # "test_dram": DramTestDataset,
    # "test_brain": BrainTestDataset,
    "test_kvasir": KvasirTestDataset,
    "test_pascal": TestDatasetPASCAL,
    "test_industrial": DatasetIndustrial,
}


def map_collate(dataset):
    return dataset.collate_fn if hasattr(dataset, "collate_fn") else None


def get_preprocessing(params):
    
    preprocess_params = params.get("preprocess", {})
    size = preprocess_params["image_size"]
    mean = preprocess_params.get("mean", "default")
    std = preprocess_params.get("std", "default")
    mean, std = get_mean_std(mean, std)
    return Compose(
            [
                Resize(size=(size, size)),
                ToTensor(),
                Normalize(mean, std),
            ]
        )
    
    
def get_dataloaders(dataset_args, dataloader_args, num_processes):
    preprocess = get_preprocessing(dataset_args)

    datasets_params = dataset_args.get("datasets")
    common_params = dataset_args.get("common", {})
    possible_batch_example_nums = dataloader_args.pop("possible_batch_example_nums")
    val_possible_batch_example_nums = dataloader_args.pop(
        "val_possible_batch_example_nums", possible_batch_example_nums
    )

    prompt_types = dataloader_args.pop("prompt_types", None)
    prompt_choice_level = dataloader_args.pop("prompt_choice_level", "batch")

    val_prompt_types = dataloader_args.pop("val_prompt_types", prompt_types)
    num_steps = dataloader_args.pop("num_steps", None)

    val_datasets_params = {
        k: v for k, v in datasets_params.items() if k.startswith("val_")
    }
    test_datasets_params = {
        k: v for k, v in datasets_params.items() if k.startswith("test_")
    }
    train_datasets_params = {
        k: v
        for k, v in datasets_params.items()
        if k not in list(val_datasets_params.keys()) + list(test_datasets_params.keys())
    }
    train_dataloader = None
    if train_datasets_params:
        train_dataset = FSSDataset(
            datasets_params=train_datasets_params,
            common_params={**common_params, "preprocess": preprocess},
        )
        train_batch_sampler = VariableBatchSampler(
            train_dataset,
            possible_batch_example_nums=possible_batch_example_nums,
            num_processes=num_processes,
            prompt_types=prompt_types,
            prompt_choice_level=prompt_choice_level,
            shuffle=True,
            num_steps=num_steps,
        )
        train_dataloader = DataLoader(
            dataset=train_dataset,
            **dataloader_args,
            collate_fn=train_dataset.collate_fn,
            batch_sampler=train_batch_sampler,
        )
    if val_datasets_params:
        val_dataloaders = {}
        for dataset, params in val_datasets_params.items():
            splits = dataset.split("_")
            if len(splits) > 2:
                dataset_name = "_".join(splits[:2])
            else:
                dataset_name = dataset
            val_dataset = FSSDataset(
                datasets_params={dataset_name: params},
                common_params={**common_params, "preprocess": preprocess},
            )
            val_batch_sampler = VariableBatchSampler(
                val_dataset,
                possible_batch_example_nums=val_possible_batch_example_nums,
                num_processes=num_processes,
                prompt_types=val_prompt_types,
            )
            val_dataloader = DataLoader(
                dataset=val_dataset,
                **dataloader_args,
                collate_fn=val_dataset.collate_fn,
                batch_sampler=val_batch_sampler,
            )
            val_dataloaders[dataset] = val_dataloader
    else:
        val_dataloaders = None
    if test_datasets_params:
        test_datasets = {
            dataset: TEST_DATASETS[dataset](**params, preprocess=preprocess)
            for dataset, params in test_datasets_params.items()
        }
        test_dataloaders = {
            name: DataLoader(
                dataset=data,
                **dataloader_args,
                collate_fn=map_collate(data),
            )
            for name, data in test_datasets.items()
        }
    else:
        test_dataloaders = None
    return (
        train_dataloader,
        val_dataloaders,
        test_dataloaders,
    )

    
def get_testloaders(dataset_args, dataloader_args):
    preprocess = get_preprocessing(dataset_args)

    datasets_params = dataset_args.get("datasets")

    test_datasets_params = {
        k: v for k, v in datasets_params.items() if k.startswith("test_")
    }

    if test_datasets_params:
        test_datasets = {
            dataset: TEST_DATASETS[dataset](**params, preprocess=preprocess)
            for dataset, params in test_datasets_params.items()
        }
        test_dataloaders = {
            name: DataLoader(
                dataset=data,
                **dataloader_args,
                collate_fn=map_collate(data),
            )
            for name, data in test_datasets.items()
        }
    else:
        test_dataloaders = None
    return test_dataloaders
