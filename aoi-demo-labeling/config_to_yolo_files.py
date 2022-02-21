# Convert the data in cvat_project_config.json to .names and .tree files for YOLO.
#
# Dong Chaoqun, April 2021, @Fraunhofer Singapore

from pathlib import Path
import argparse
import json

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="./configs/cvat_project_config.json", help="Path to CVAT configuration file.")    
    parser.add_argument("--output_path", default="", help="Path for output file.")
    parser.add_argument(
        "--multilabel", default=False, help="Whether support multilabel: element class + defect class.")

    args = parser.parse_args()
    
    input_path = Path(args.input_path)  
    multilabel = args.multilabel
    if len(args.output_path) == 0 and multilabel:
        output_path = Path("./yolo_hierarchy_multi")
    elif len(args.output_path) == 0 and not multilabel:
        output_path = Path("./yolo_hierarchy")
    else:
        output_path = Path(args.output_path)
    if not output_path.exists():
        output_path.mkdir()

    try:
        with open(input_path) as f:
            config_data = json.load(f)
    except IOError:
        print("[Error]: cannot open CVAT configuration file.")
        return False
    else:
        f.close()

    # get element classes including subclasses
    parents = []
    children = []
    tree_p = []
    tree_c = []
    for i, element_entry in enumerate(config_data):
        element = element_entry["name"]
        parents.append(element)
        tree_p.append(-1)
        if "attributes" in element_entry:
            attris = element_entry["attributes"]
            for attri in attris:
                if attri["name"] == "subClass":
                    vals = attri["values"]
                    for val in vals:
                        children.append(val)
                        tree_c.append(i)
                    break
    tree = tree_p + tree_c

    # get defect classes
    if multilabel:
        defects = []
        for element_entry in config_data:
            if "attributes" in element_entry:
                attris = element_entry["attributes"]
                for attri in attris:
                    if attri["name"] == "defectClass":
                        values = attri["values"]
                        for defect in values:
                            defects.append(defect)
                            tree.append(-1)
            break  
        names = parents + children + defects
    else:
        names = parents + children

    # write file
    name_file_path = output_path / "obj.names"
    tree_file_path = output_path / "obj.tree"

    with open(name_file_path, 'w') as f:
        for name in names:
            f.write(f"{name}\n")
    f.close()

    with open(tree_file_path, 'w') as f:
        for i, node in enumerate(tree):
            f.write(f"{i} {node}\n")
    f.close()

    if multilabel:
        print(f"Created .name and .tree file for multiclass.")
    else:
        print(f"Created .name and .tree file.")

if __name__ == "__main__":
    main()