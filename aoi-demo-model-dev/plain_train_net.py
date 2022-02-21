#!/usr/bin/env python
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.
"""

import logging
import argparse
from pathlib import Path
import os
import json
import copy
import numpy as np
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import detection_utils
from detectron2.data import transforms as T
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.data.dataset_mapper import DatasetMapper

from detectron2.structures import (
    Boxes,
    BoxMode,
    Instances
)
from detectron2.engine import default_setup, default_writers, launch
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage

from src.config import add_sub_head_config
from src.data.datasets import (
    register_custom_coco_instances,
    get_meta_info
)

logger = logging.getLogger("detectron2")


class CustomDatasetMapper(DatasetMapper):

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(
            dataset_dict)  # it will be modified by code below

        image = detection_utils.read_image(
            dataset_dict["file_name"], format=self.image_format)
        detection_utils.check_image_size(dataset_dict, image)

        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                detection_utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = detection_utils.filter_empty_instances(
                instances)
        return dataset_dict


def annotations_to_instances(annos, image_size, mask_format="polygon"):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes", "gt_sub_classes", "gt_defect_classes"
            "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = [BoxMode.convert(
        obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)

    classes = [int(obj["category_id"]) for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    sub_classes = [int(obj["attributes"]["subClass"])
                   if "subClass" in obj["attributes"] else -1 for obj in annos]
    sub_classes = torch.tensor(sub_classes, dtype=torch.int64)
    target.gt_sub_classes = sub_classes

    defect_classes = [int(obj["attributes"]["defectClass"])
                      if "defectClass" in obj["attributes"] else -1 for obj in annos]
    defect_classes = torch.tensor(defect_classes, dtype=torch.int64)
    target.gt_defect_classes = defect_classes

    return target


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = [
        COCOEvaluator(dataset_name, output_dir=output_folder)
    ]

    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(
                cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info(
                "Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]

    return results


def do_train(cfg, model, resume=False):
    best_score = 0.0
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(
            cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = default_writers(
        cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    custom_mapper = CustomDatasetMapper(cfg, True)
    data_loader = build_detection_train_loader(cfg, mapper=custom_mapper)

    # compared to "train_net.py", accurate timing and precise BN are not supported here
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item()
                                 for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(
                    total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar(
                "lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                eval_result = do_test(cfg, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

                eval_ap = eval_result['bbox']['AP']
                if eval_ap > best_score:
                    best_score = eval_ap
                    checkpointer.save("best_model", iteration=iteration)
                    logger.info(
                        f"Saving new best model. Validation AP: {eval_ap}")

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


def setup(args):
    """
    Create configs and perform basic setups.
    """

    # setup train/test data
    data_path = Path(args.data_path)
    data_config_path = data_path / 'cvat_project_config.json'
    train_ann_path = data_path / 'train_instances_coco.json'
    test_ann_path = data_path / 'test_instances_coco.json'
    sub_class_list = ['subClass', 'defectClass']

    meta_dict = get_meta_info(
        train_ann_path, data_config_path, sub_class_list)

    register_custom_coco_instances(
        "aoi_train", meta_dict, train_ann_path, data_path, sub_class_list)
    register_custom_coco_instances(
        "aoi_test", meta_dict, test_ann_path, data_path, sub_class_list)

    cfg = get_cfg()
    add_sub_head_config(cfg)

    if len(args.config_file) > 0:
        cfg.merge_from_file(args.config_file)
    else:
        # default params
        config_file_name = "./configs/detectron2_faster_rcnn_R_50_FPN_3x_SubHeads.yaml"

        cfg.merge_from_file(config_file_name)
        cfg.OUTPUT_DIR = f"./logs/{Path(config_file_name).stem}"
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(meta_dict['thing_classes'])

        # [Resistor, LED, Electrolytic Capacitor, Ceramic Capacitor, Wire]
        # cfg.MODEL.ROI_HEADS.SUB_HEAD.NUM_CLASSES = [10, 5, 2, 2, 7]

        cfg.SOLVER.MAX_ITER = 3010
        cfg.TEST.EVAL_PERIOD = 100

    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # logging etc

    with open(Path(cfg.OUTPUT_DIR) / "data_meta_info.json", "w") as meta_file:
        json.dump(meta_dict, meta_file, indent=4, sort_keys=True)    

    return cfg


def main(args):
    cfg = setup(args)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg, model, resume=args.resume)
    return do_test(cfg, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="",
                        metavar="PATH", help="path to root of train data")
    parser.add_argument("--config-file", default="",
                        metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--eval-only", action="store_true",
                        help="perform evaluation only")
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
        "See config references at "
        "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        1,  # num_gpus
        args=(args,),
    )
