import os
import os.path as p
import re
import shutil
from argparse import ArgumentParser


def restore_original_name(root_dir, find, replace):
    """Restore original file name, needs modification if dataset is changed"""
    for file in sorted(os.listdir(root_dir)):
        fixed_name = file
        fixed_name = fixed_name.replace("portai20211215111931_", "")
        fixed_name = re.sub("_detection_[0-9]*", "", fixed_name)
        fixed_name_split = fixed_name.split("_")
        fixed_name = ""
        for i in range(len(fixed_name_split)):
            if i == 0:
                fixed_name = fixed_name_split[i] + "_"
            elif i < len(fixed_name_split) - 1:
                fixed_name += fixed_name_split[i] + " "
            else:
                fixed_name += fixed_name_split[i]

        original_path = p.join(root_dir, file)
        renamed_path = p.join(root_dir, fixed_name.replace(find, replace))
        shutil.move(original_path, renamed_path)


def find_replace(root_dir, find, replace):
    """Rename each file by find and replace pattern, rename done by moving file in same directory."""
    for file in sorted(os.listdir(root_dir)):
        original_path = p.join(root_dir, file)
        renamed_path = p.join(root_dir, original_path.replace(find, replace))
        shutil.move(original_path, renamed_path)


if __name__ == "__main__":
    parser = ArgumentParser("Rename modified files back to original")
    parser.add_argument("root_dir", help="root folders, all files should be in there")
    parser.add_argument(
        "-f",
        help="Find this string , replace with 'r' parameter",
    )
    parser.add_argument("-r", help="Replace with....")
    parser.add_argument("-mode", choices=["find_replace", "restore"], required=True)
    args = parser.parse_args()
    if args.mode == "find_replace":
        find_replace(args.root_dir, args.f, args.r)
    if args.mode == "restore":
        restore_original_name(args.root_dir, args.f, args.r)
