from affex.data import get_dataloaders
from affex.data.utils import BatchKeys
from affex.utils.logger import get_logger
from affex.utils.utils import load_yaml


import pandas as pd
from tqdm import tqdm


import copy
import os


def generate_dataset(parameters):
    parameters = load_yaml(parameters)
    assert "dataset" in parameters, "Parameters file must contain a 'dataset' key"

    outfolder = parameters["output_dir"]
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    _, val, _ = get_dataloaders(
        copy.deepcopy(parameters["dataset"]),
        copy.deepcopy(parameters["dataloader"]),
        num_processes=1,
    )
    datasets = {}
    logger = get_logger("Dataset Generation")

    for dataset_name, val_dataloader in val.items():
        logger.info(f"Creating {dataset_name}")
        dataset = pd.DataFrame(columns=[BatchKeys.IMAGE_IDS.value, BatchKeys.CLASSES.value])
        bar = tqdm(
            val_dataloader,
            total=len(val_dataloader),
            desc=f"Creating dataset {dataset_name}",
            unit="batch",
        )
        for batch in bar:
            batch, _ = batch
            batch, gt = batch
            image_ids = batch[BatchKeys.IMAGE_IDS]
            class_ids = batch[BatchKeys.CLASSES]
            batch_df = pd.DataFrame(
                {
                    BatchKeys.IMAGE_IDS.value: [image_ids],
                    BatchKeys.CLASSES.value: [class_ids],
                }
            )
            dataset = pd.concat([dataset, batch_df], ignore_index=True)
        logger.info(f"Dataset {dataset_name} created")
        datasets[dataset_name] = dataset

    for dataset_name, dataset in datasets.items():
        dataset_file = os.path.join(outfolder, f"{dataset_name}.csv")
        dataset.to_csv(dataset_file, index=False)
        logger.info(f"Dataset {dataset_name} saved to {dataset_file}")