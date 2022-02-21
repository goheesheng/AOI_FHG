from pathlib import Path
import argparse
import json
import copy


def update_annotation_ids(anns):
    for i, ann in enumerate(anns):
        ann["id"] = i+1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="./data/",
                        help="Path to folder containing images and their annotations in a json file in COCO format. This folder will become the root folder of the combined annotations.")
    parser.add_argument("--output_name", default="combined_instances_coco.json",
                        help="Name for output file which contains the combined annotations. It will be located in input_path.")

    args = parser.parse_args()

    input_path = Path(args.input_path)
    ann_name_pattern = "instances_*.json"

    if len(args.output_name) > 0:
        output_path = input_path / args.output_name
    else:
        output_path = input_path / "combined_instances_coco.json"

    input_ann_paths = list(input_path.glob(f"**/{ann_name_pattern}"))

    if len(input_ann_paths) == 0:
        print(
            f"No coco annotations found in input folder (expected file name '{ann_name_pattern}'")
        return

    # copy meta info from first file
    coco_out_data = {}
    with open(input_ann_paths[0], 'r') as f:
        coco_data = json.load(f)
    coco_out_data = coco_data.copy()
    coco_out_data["images"] = []
    coco_out_data["annotations"] = []

    current_image_id = 1

    for input_ann_path in input_ann_paths:

        rel_path = input_ann_path.relative_to(input_path).parent        

        with open(input_ann_path, 'r') as f:
            coco_data = json.load(f)
            images = coco_data["images"]
            annotations = coco_data["annotations"]
            images = copy.deepcopy(images)
            annotations = copy.deepcopy(annotations)

        # adjust folder paths and image ids in images and annotations
        for image in images:
            image["file_name"] = str(rel_path / image["file_name"])
            image_anns = [
                ann for ann in annotations if ann["image_id"] == image["id"]]
            image["id"] = current_image_id
            for image_ann in image_anns:
                image_ann["image_id"] = current_image_id
            current_image_id += 1

        coco_out_data["images"].extend(images)
        coco_out_data["annotations"].extend(annotations)

    update_annotation_ids(coco_out_data["annotations"])

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    with open(output_path, 'w') as out:
        json.dump(coco_out_data, out, indent=4, sort_keys=True)

    print("Done")


if __name__ == "__main__":
    main()
