import argparse
from pathlib import Path
import numpy as np
from PIL import Image

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="./data/chip_id", help="Path to folder containing images of a chip setup.")    
    parser.add_argument("--output_path", default="", help="Path for output folder. Same as input folder if empty. Warning: this will override existing images.")    

    args = parser.parse_args()
    
    input_path = Path(args.input_path)  

    if len(args.output_path) > 0:
        output_path = Path(args.output_path)
    else:
        output_path = input_path

    if not output_path.exists():
        output_path.mkdir(parents=True)

    image_paths = input_path.glob('*.jpg')

    n_rotated = 0

    for image_path in image_paths:
        img = Image.open(image_path)

        if img.height < img.width:
            img = img.transpose(method=Image.ROTATE_270)
            n_rotated += 1
            
        # Save anyway because the pillow encoding seems to take less memory than my phone
        output_img_path = output_path / image_path.name
        img.save(output_img_path, "JPEG")

    print(f"Done. {n_rotated} images rotated")
