from pathlib import Path
import time
import argparse
import pickle
import matplotlib.pyplot as plt
import cv2
import torch

from src.data.utils import xyxy_to_xywh
from src.visualization.visualize import plot_img_with_anns


def pre_processing(img, input_format, target_shape):
    if input_format == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_shape, interpolation=cv2.INTER_AREA)

    return img


def inference(image, target_shape, thing_classes):

    # pre-processing
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    inputs = {"image": image,
              "height": target_shape[1], "width": target_shape[0]}

    with torch.no_grad():
        outputs = model([inputs])[0]

    # post-processing
    instances = outputs["instances"].to("cpu")

    pred_bboxes = instances.pred_boxes.tensor.numpy()
    pred_bboxes = map(xyxy_to_xywh, pred_bboxes)

    pred_classes = instances.pred_classes.numpy()

    pred_anns = []
    for bbox, cat_id in zip(pred_bboxes, pred_classes):
        pred_ann = {
            'bbox': list(bbox),
            'category_id': cat_id,
            'bbox_mode': 'XYWH_ABS',
            'category_name': thing_classes[cat_id]
        }
        pred_anns.append(pred_ann)

    return pred_anns


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", default="./model/model_name", help="Path to the model file.")
    parser.add_argument("--image_path", default="",
                        help="Path to the image file")

    args = parser.parse_args()

    model_path = Path(args.model_path)

    if len(args.image_path) > 0:
        img_path = Path(args.image_path)
    else:
        img_path = None

    target_shape = (768, 1024)  # width, height (because openCV)

    # load model
    with open(model_path, "rb") as f:
        state_dict = pickle.load(f)
        model = state_dict["model"]
        input_format = state_dict["input_format"]
        thing_classes = state_dict["thing_classes"]

    model.eval()
    print(f"Running on: {next(model.parameters()).device}")

    if img_path is None:  # video mode
        cap = cv2.VideoCapture(0)
        start = time.time()
        fig, ax = plt.subplots(figsize=(28, 21))
        plt.ion()
        plt.show()
        while(time.time()-start < 30):
            # capture frame-by-frame
            ret, frame = cap.read()
            if frame.shape[0] < frame.shape[1]:
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
            frame = pre_processing(frame, input_format, target_shape)
            pred_anns = inference(frame, target_shape, thing_classes)
            # display results
            ax.clear()
            _ = plot_img_with_anns(frame, None, pred_anns, ax=ax)
            plt.pause(0.001)

        cap.release()
        print("Done")

    else:  # single image mode
        img = cv2.imread(str(img_path))

        start = time.time()
        img = pre_processing(img, input_format, target_shape)
        pred_anns = inference(img, target_shape, thing_classes)
        duration = (time.time() - start)
        print(f"Inference took: {duration:.3f} s")

        # Plot
        img_name = img_path.stem
        chip_name = img_path.parent.stem
        ax = plot_img_with_anns(img, None, pred_anns, figsize=(
            28, 21), title=f"Predictions for Chip {chip_name}, Image {img_name}")
        plt.show()
