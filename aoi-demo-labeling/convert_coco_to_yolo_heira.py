# Convert coco annotation to yolo format with hierarchy.
#
# If hierarchical classes, only label the object with the end child class.
# Support multiclass labeling with both element class and defect class.
# Label files are saved in the same folder as the images.
#
# Dong Chaoqun, April 2021, @Fraunhofer Singapore

from pathlib import Path
import argparse
import json
import numpy as np


def load_tree(tree_file_path):
    try:
        with open(tree_file_path, 'r') as t_f:
            flat_tree = [int(l[:-1].split(' ')[1]) for l in t_f.readlines()]
    except IOError:
        print("[ERROR]: cannot open .tree file. Run config_to_yolo_files.py first.")
        return False
    else:
        t_f.close()
    return flat_tree


def load_names(name_file_path):
    try:
        with open(name_file_path, 'r') as f:
            names = [l[:-1] for l in f.readlines()]
    except IOError:
        print("[ERROR]: cannot open .names file. Run config_to_yolo_files.py first.")
        return False
    else:
        f.close()
    return names


def load_coco_anno_data(sub_folder_path):
    coco_anno_json_path = sub_folder_path / 'instances_default_expanded.json'   # hard code the JSON file name
    try:
        with open(coco_anno_json_path) as f:
            coco_anno_data = json.load(f)
    except IOError:
        print("[ERROR]: cannot open COCO annotation JSON file.")
        return False
    else:
        f.close()
    return coco_anno_data


def get_images_info(image_data):
    images_info_dict = {}
    for img in image_data:
        if img["id"] not in images_info_dict:
            images_info_dict[img["id"]] = [img["file_name"].split('.')[0], img["width"], img["height"]]
    return images_info_dict


def get_cate_info(category):
    cate_info_dict= {}
    for cate in category:
        if cate["id"] not in cate_info_dict:
            cate_info_dict[cate["id"]]= cate["name"]
    return cate_info_dict


def get_hiera_class_name(anno, cate_info_dict):
    # no defect class
    hiera_class_name = [] 
    if "attributes" not in anno:  # actually should always have attributes for storing defect class
        hiera_class_name.append(cate_info_dict[anno["category_id"]])
    elif "subClass" in anno["attributes"]:
        hiera_class_name.append(anno["attributes"]["subClass"])
    else:
        hiera_class_name.append(cate_info_dict[anno["category_id"]])
    return hiera_class_name


def get_hiera_multi_class_name(anno, cate_info_dict):
     # support multiple labels
    hiera_class_name = [] 
    if "attributes" not in anno:
        hiera_class_name.append(cate_info_dict[anno["category_id"]])
    else:
        if "defectClass" in anno["attributes"]:
            hiera_class_name.append(anno["attributes"]["defectClass"])
        if "subClass" in anno["attributes"]:
            hiera_class_name.append(anno["attributes"]["subClass"])
        else:
            hiera_class_name.append(cate_info_dict[anno["category_id"]])
    return hiera_class_name

def get_labels_with_tree(anno, cate_info_dict, tree, names):
    # to deal with cases where subclasses from different parent classes share same name
    labels = []
    if "attributes" not in anno:
        # no attributes means no subclass no defect class, add element class label
        labels.append(names.index(cate_info_dict[anno["category_id"]]))
    else:
        if "defectClass" in anno["attributes"]:
            # have defect class, add defect class label
            labels.append(names.index(anno["attributes"]["defectClass"]))
        if "subClass" in anno["attributes"]:
            # have subclass check its parent class the add its label
            sub_cls_name = anno["attributes"]["subClass"]
            ele_cls_name = cate_info_dict[anno["category_id"]]
            ele_idx = names.index(ele_cls_name)
            sub_idx = [i for i,x in enumerate(names) if x==sub_cls_name]
            for i in sub_idx:
                pi = tree[i]
                if pi == ele_idx:
                    labels.append(i)
                    break 
        else:
            labels.append(names.index(cate_info_dict[anno["category_id"]]))
    return labels


def class_name_to_label(names_list, class_name):
    labels = []
    for name in class_name:
        labels.append(names_list.index(name))
    return labels


def convert_to_yolo_box(img_w, img_h, x1, y1, x2, y2):
    '''Converts (x1, y1, x1, y2) KITTI format to
    (x, y, width, height) normalized YOLO format'''
    def sorting(l1, l2):
        if l1 > l2:
            lmax, lmin = l1, l2
            return lmax, lmin
        else:
            lmax, lmin = l2, l1
            return lmax, lmin
    xmax, xmin = sorting(x1, x2)
    ymax, ymin = sorting(y1, y2)
    dw = 1./img_w
    dh = 1./img_h
    x = (xmin + xmax)/2.0
    y = (ymin + ymax)/2.0
    w = xmax - xmin
    h = ymax - ymin
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return x,y,w,h


def main():
    parser= argparse.ArgumentParser()
    parser.add_argument(
        "--tree_file_path", default="./yolo_hierarchy_multi/obj.tree", help="Path to YOLO .tree file.")
    parser.add_argument(
        "--names_file_path", default="./yolo_hierarchy_multi/obj.names", help="Path to YOLO .tree file.")
    parser.add_argument("--data_path", default="../data/data_1_2",
                        help="Path to data folder where each subfolder contains images and the JSON file storing COCO annotations.")
    parser.add_argument(
        "--multilabel", default=True, help="Whether support multilabel: element class + defect class.")

    args= parser.parse_args()
    tree_file_path= Path(args.tree_file_path)
    names_file_path= Path(args.names_file_path)
    data_path= Path(args.data_path)
    multilabel = args.multilabel 

    # get name and tree data
    tree = load_tree(tree_file_path)
    names= load_names(names_file_path)
    assert len(tree) == len(
        names), ".tree file and .names file should contain the same number of entires."

    # get all subfolders
    sub_folders = list(data_path.glob("*/"))
    assert len(sub_folders) != 0, "Cannot find any subfolders."

    for sub_folder_path in sub_folders:
        # get coco annotation data
        coco_anno_data = load_coco_anno_data(sub_folder_path)

        anno_data= coco_anno_data["annotations"]
        image_data= coco_anno_data["images"]
        cate_data= coco_anno_data["categories"]

        image_info_dict= get_images_info(image_data)
        cate_info_dict= get_cate_info(cate_data)

        img_txt_set = set()

        for anno in anno_data:
            # get image info
            img_id= anno["image_id"]
            img_file_name, img_w, img_h = image_info_dict[img_id]
            img_txt_file_name = img_file_name + '.txt'

            # get hierarchical label
            if multilabel:
                # hiera_class_name = get_hiera_multi_class_name(anno, cate_info_dict)
                labels = get_labels_with_tree(anno, cate_info_dict, tree, names)
            else:
                hiera_class_name = get_hiera_class_name(anno, cate_info_dict)
                labels = class_name_to_label(names, hiera_class_name)  # support multiple labels

            # convert bbox
            bbox = anno["bbox"]
            kitti_bbox = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
            b_x, b_y, b_w, b_h = convert_to_yolo_box(img_w, img_h, kitti_bbox[0], kitti_bbox[1], kitti_bbox[2], kitti_bbox[3])

            # write label and bbox to image .txt file
            for label in labels:
                content = f"{label} {b_x:.6f} {b_y:.6f} {b_w:.6f} {b_h:.6f}"
                if img_id not in img_txt_set:
                    img_txt_set.add(img_id)               
                    img_txt_file = open(sub_folder_path / img_txt_file_name, 'w')
                elif img_id in img_txt_set:
                    img_txt_file = open(sub_folder_path / img_txt_file_name, 'a')
                    img_txt_file.write("\n")
                img_txt_file.write(content)
                img_txt_file.close()

    if multilabel:
         print("Created multi-label in YOLO format.")
    else:
        print("Created labels in YOLO format.")


if __name__ == "__main__":
    main()
