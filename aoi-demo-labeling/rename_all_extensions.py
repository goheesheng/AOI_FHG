from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="./data/",
                        help="Path to folder which should be processed.")
    parser.add_argument("--original_ext", default="JPG",
                        help="Extension which should be renamed")
    parser.add_argument("--target_ext", default="jpg",
                        help="New extension name")                        

    args = parser.parse_args()

    input_path = Path(args.input_path)
    org_ext = args.original_ext
    tgt_ext = args.target_ext

    file_pattern = f"*.{org_ext}"

    file_paths = list(input_path.glob(f"**/{file_pattern}"))

    if len(file_paths) == 0:
        print(
            f"No files for renaming found (pattern '{file_pattern}'")
        return

    n_renamed = 0
    for file_path in file_paths:        
        new_file_path = file_path.with_suffix(f".{tgt_ext}")
        file_path.rename(new_file_path)      
        n_renamed += 1

    print(f"Done. {n_renamed} files renamed")


if __name__ == "__main__":
    main()
