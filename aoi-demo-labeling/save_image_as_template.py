from pathlib import Path
import shutil
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="./data/combined_instances_coco.json",
                        help="Path to COCO annotations")
    parser.add_argument("--image_name", default="IMG_20210301_124858.jpg",
                        help="Name of the image which will be saved as template")
    parser.add_argument("--output_path", default="./",
                        help="Path to folder in which annotations and image will be stored")
    parser.add_argument("--output_id", default=1, type=int,
                        help="The id under which the template will be stored. Will be used also for the output file names.")

    args = parser.parse_args()

    input_path = Path(args.input_path)
    image_name = args.image_name

    with open(input_path, "r") as f:
        ann_json = json.load(f)

    categories = ann_json['categories']
    category_dict = {c["id"]: c["name"] for c in categories}

    # find image in coco annotations
    img_dict = next((im for im in ann_json['images']
                     if image_name in im['file_name']), None)

    if img_dict is None:
        print(f"Image {image_name} was not found")
        return

    # collect annotations
    img_id = img_dict["id"]
    img_anns_raw = [ann for ann in ann_json['annotations']
                    if ann['image_id'] == img_id]

    img_anns = []
    for ann in img_anns_raw:
        img_ann = {
            'bbox': ann['bbox'],
            'category_id': ann['category_id'],
            'name': category_dict[ann['category_id']],
            'defectClass': ann['attributes']['defectClass']
        }
        sub_class = ann['attributes'].get('subClass', None)
        if sub_class is not None:
            img_ann['subClass'] = sub_class

        img_anns.append(img_ann)

    # write output files
    output_path = Path(args.output_path)
    output_id = args.output_id
    output_name = f"{str(output_id)}{Path(image_name).suffix}"

    output_dict = {
        "image": output_name,
        "relPath": f"templates/{output_name}",
        "name": Path(output_name).stem,
        "dimensions": {
            "width": img_dict["width"],
            "height": img_dict["height"]
        },
        "id": output_id,
        "annotations": img_anns
    }

    if not output_path.exists():
        output_path.mkdir(parents=True)

    out_json_path = output_path / Path(output_name).with_suffix(".json")

    with open(out_json_path, 'w') as outfile:
        json.dump(output_dict, outfile, indent=4)

    img_path = input_path.parent / img_dict["file_name"]
    out_img_path = output_path / output_name

    shutil.copy(img_path, out_img_path)

    print(
        f"Done. Saved annotations to {out_json_path} and image to {out_img_path}")


if __name__ == "__main__":
    main()
