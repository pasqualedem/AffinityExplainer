import json
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--instances_path')
    return parser.parse_args()


def rename_coco20i_json(instances_path: str):
    """Change image filenames of COCO 2014 instances.

    Args:
        instances_path (str): Path to the COCO 2014 instances file.
    """
    with open(instances_path, "r") as f:
        anns = json.load(f)
    for image in anns["images"]:
        image["file_name"] = image["file_name"].split("_")[-1]
    with open(instances_path, "w") as f:
        json.dump(anns, f)


if __name__ == '__main__':
    args = parse_args()
    assert args.instances_path.endswith('.json'), "instances must be a json file."
    rename_coco20i_json(args.instances_path)