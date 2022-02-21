from pathlib import Path
import argparse
import json
import copy
import random
import math


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="./data/combined_instances_coco.json",
                        help="Path to a COCO annotations which shall be split.")
    parser.add_argument("--test_frac", default=0.33, type=float,
                        help="Fraction of data which will be used for the test set")

    args = parser.parse_args()

    input_path = Path(args.input_path)
    test_frac = args.test_frac

    with open(input_path, 'r') as f:
        input_data = json.load(f)

    # chip ids are determined by the relative image paths compared to root
    img_chip_ids = [
        str(Path(img["file_name"]).parent) for img in input_data["images"]]
    img_chip_ids = list(set(img_chip_ids))

    # split chip ids in train/test
    random.shuffle(img_chip_ids)

    n_test = math.ceil(test_frac * len(img_chip_ids))
    test_ids = img_chip_ids[:n_test]
    train_ids = img_chip_ids[n_test:]

    train_data = select_data_by_subfolder_name(input_data, train_ids)
    test_data = select_data_by_subfolder_name(input_data, test_ids)

    update_annotation_ids(train_data["annotations"])
    update_annotation_ids(test_data["annotations"])

    output_train_path = input_path.parent / "train_instances_coco.json"
    output_test_path = input_path.parent / "test_instances_coco.json"

    with open(output_train_path, 'w') as out:
        json.dump(train_data, out, indent=4, sort_keys=True)

    with open(output_test_path, 'w') as out:
        json.dump(test_data, out, indent=4, sort_keys=True)

    print("Done")


def select_data_by_subfolder_name(data, ids):
    selected_data = data.copy()

    selected_data["images"] = [img for img in data["images"]
                               if str(Path(img["file_name"]).parent) in ids]

    selected_img_ids = [img["id"] for img in selected_data["images"]]
    selected_data["annotations"] = [
        ann for ann in data["annotations"] if ann["image_id"] in selected_img_ids]

    return selected_data


def update_annotation_ids(anns):
    for i, ann in enumerate(anns):
        ann["id"] = i+1


if __name__ == "__main__":
    main()
