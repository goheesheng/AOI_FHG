from pathlib import Path
import argparse
import json
import copy
import numpy as np
import cv2

from src.registration import register_images_fm
from src.utils import xyxy_to_xywh, xywh_xyxy


def register_annotations(anns, image_info, template_info, input_path):

    # load images
    data_path = input_path.parent
    img_path = data_path / image_info["file_name"]
    templ_path = data_path / template_info["file_name"]
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    templ_img = cv2.imread(str(templ_path), cv2.IMREAD_GRAYSCALE)
    
    mat = register_images_fm(img, templ_img)

    if mat is None:
        print(f"Registration failed for image {image_info['file_name']}. Template bounding boxes are kept")
        return

    bboxes = [ann["bbox"] for ann in anns]

    # convert bounding boxes to list of points
    bboxes_xyxy = np.array([xywh_xyxy(box) for box in bboxes])
    p1s = bboxes_xyxy[:,:2]
    p2s = bboxes_xyxy[:,2:]
    ps = np.vstack((p1s, p2s))
  
    # transform points according to registration matrix
    pers_mat = np.zeros(shape=(3, 3))
    pers_mat[:2,:] = mat
    pers_mat[2] = [0, 0, 1]
    ps_reg = cv2.perspectiveTransform(np.array([ps]), pers_mat)[0]            

    # convert points back to bounding boxes in XYWH format
    p1s_reg = ps_reg[:len(bboxes)]
    p2s_reg = ps_reg[len(bboxes):]
    bboxes_xyxy_reg = np.hstack((p1s_reg, p2s_reg)).tolist()    
    bboxes_reg = [xyxy_to_xywh(box) for box in bboxes_xyxy_reg]

    # update annotations
    for bbox, ann in zip(bboxes_reg, anns):
        ann["bbox"] = bbox
        w, h = bbox[2:]
        ann["area"] = w * h

    return


def update_image_id(anns, image_id):
    for ann in anns:
        ann["image_id"] = image_id


def update_annotation_ids(anns):
    for i, ann in enumerate(anns):
        ann["id"] = i+1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="./data/instances_default.json", help="Path to single image labels in COCO 1.0 format.")    
    parser.add_argument("--output_path", default="", help="Path for output file.")
    parser.add_argument('--register', default=False, action='store_true')

    args = parser.parse_args()
    
    input_path = Path(args.input_path)  

    if len(args.output_path) > 0:
        output_path = Path(args.output_path)
    else:
        output_path = input_path.parent / f"{input_path.stem}_expanded{input_path.suffix}"

    with open(input_path) as f:
        coco_data = json.load(f)      

    coco_out_data = coco_data.copy()

    images = coco_out_data["images"]
    annotations = coco_out_data["annotations"]
    anns_template = copy.deepcopy(annotations)

    ann_image_ids = [ann["image_id"] for ann in annotations]
    ann_image_ids = list(set(ann_image_ids)) # find unique ids

    if len(ann_image_ids) > 1:
        raise ValueError("Input annotations contain labels for more than 1 image. Delete all annotations except for a single image.")

    ann_image_id = ann_image_ids[0]
    other_image_ids = [image["id"] for image in images if image["id"] != ann_image_id]    

    ann_image = [image for image in images if image["id"] == ann_image_id][0]

    for image_id in other_image_ids:
        new_anns = copy.deepcopy(anns_template)
        update_image_id(new_anns, image_id)
        
        if args.register:
            image = [im for im in images if im["id"] == image_id][0]
            register_annotations(new_anns, image, ann_image, input_path)

        annotations.extend(new_anns)     
        
    update_annotation_ids(annotations)

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    with open(output_path, 'w') as out:
        json.dump(coco_out_data, out, indent=4, sort_keys=True)

    print("Done")