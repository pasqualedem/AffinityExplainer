import pandas as pd

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from affex.data.dataset import FSSDataset, VariableBatchSampler
from affex.data.transforms import Normalize, Resize
from affex.data.utils import get_mean_std


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
    """One dataloader per evaluation set named in the parameters.

    Episodes come from the fixed lists in ``csv_folder`` when one is given, so every run
    sees the same support/query pairs. ``num_processes``/``process_id`` keep one chunk of
    that list, which is how a long grid is split across jobs.
    """
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
        batch_size = dataloader_args.pop("batch_size")
        possible_batch_example_nums = [[batch_size]]
        val_possible_batch_example_nums = [[batch_size]]

    dataloader_args.pop("num_steps", None)

    dataloaders = {}
    for dataset, params in datasets_params.items():
        splits = dataset.split("_")
        dataset_name = "_".join(splits[:2]) if len(splits) > 2 else dataset
        val_dataset = FSSDataset(
            datasets_params={dataset_name: params},
            common_params={**common_params, "preprocess": preprocess},
        )
        if csv_folder is not None:
            episodes = pd.read_csv(f"{csv_folder}/{dataset}.csv")
            episodes = episodes.applymap(eval)

            if process_id is not None and num_split_processes is not None:
                episodes = episodes[episodes.index % num_split_processes == process_id]
                episodes.reset_index(drop=True, inplace=True)
        else:
            episodes = None

        batch_sampler = VariableBatchSampler(
            val_dataset,
            possible_batch_example_nums=val_possible_batch_example_nums,
            num_processes=num_processes,
            metadata_df=episodes,
        )
        dataloaders[dataset] = DataLoader(
            dataset=val_dataset,
            **dataloader_args,
            collate_fn=val_dataset.collate_fn,
            batch_sampler=batch_sampler,
        )
    return dataloaders
