import pandas as pd
import torch

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from affex.data.dataset import FSSDataset, VariableBatchSampler
from affex.data.deepglobe import DatasetDeepglobe
from affex.data.evican import EVICAN
from affex.data.industrial import DatasetIndustrial
from affex.data.isic import DatasetISIC
from affex.data.kvasir import KvasirTestDataset
from affex.data.lab2wild import Lab2Wild
from affex.data.lung import DatasetLung
from affex.data.nucleus import Nucleus
from affex.data.test_pascal import TestDatasetPASCAL
from affex.data.pothole import Pothole
from affex.data.transforms import Normalize, Resize

from affex.data.coco import CocoLVISDataset
from affex.data.coco_crop import CocoLVISCrop
from affex.data.utils import get_mean_std

from affex.data.weedmap import WeedMapTestDataset


TEST_DATASETS = {
    "test_weedmap": WeedMapTestDataset,
    "test_deepglobe": DatasetDeepglobe,
    "test_isic": DatasetISIC,
    "test_evican": EVICAN,
    "test_nucleus": Nucleus,
    "test_pothole": Pothole,
    "test_lab2wild": Lab2Wild,
    "test_lungcancer": DatasetLung,
    "test_kvasir": KvasirTestDataset,
    "test_pascal": TestDatasetPASCAL,
    "test_industrial": DatasetIndustrial,
}


def map_collate(dataset):
    return dataset.collate_fn if hasattr(dataset, "collate_fn") else None


def get_preprocessing(params):
    
    preprocess_params = params.get("preprocess", {})
    size = preprocess_params.get("image_size", 256)
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
    possible_batch_example_nums = dataloader_args.pop("possible_batch_example_nums", None)
    val_possible_batch_example_nums = dataloader_args.pop(
        "val_possible_batch_example_nums", possible_batch_example_nums
    )
    num_split_processes = dataloader_args.pop("num_processes", None)
    process_id = dataloader_args.pop("process_id", None)
    csv_folder = dataloader_args.pop("csv_folder", None)
    
    if "batch_size" in dataloader_args:
        batch_size = dataloader_args["batch_size"]
        dataloader_args.pop("batch_size")
        possible_batch_example_nums = [[batch_size]]
        val_possible_batch_example_nums = [[batch_size]]

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
            if csv_folder is not None:
                val_dataset_csv = pd.read_csv(f"{csv_folder}/{dataset}.csv")
                val_dataset_csv = val_dataset_csv.applymap(eval)
                
                if process_id is not None and num_split_processes is not None:
                    # keep only the rows for the current process, row_idx % num_processes == process_idx
                    val_dataset_csv = val_dataset_csv[
                        val_dataset_csv.index % num_split_processes == process_id
                    ]
                    # reset index
                    val_dataset_csv.reset_index(drop=True, inplace=True)
            else:
                val_dataset_csv = None
            
            val_batch_sampler = VariableBatchSampler(
                val_dataset,
                possible_batch_example_nums=val_possible_batch_example_nums,
                num_processes=num_processes,
                metadata_df=val_dataset_csv,
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
