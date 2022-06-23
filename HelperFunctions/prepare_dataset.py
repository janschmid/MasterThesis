"""The purpose of this class is to make it as easy as possible with only the source code to start with the project on a new machine.
This class downloads the dataset and perpares it for usage. 
Because not the entire code is public, it can be neccessary to add repositories to the parent folder of this repository"""
import os
import os.path as p
import shutil

import kitti_format_handler

file_path = os.path.dirname(os.path.realpath(__file__))
from argparse import ArgumentParser

import separate_broken_samples

# Download custom dataset, restricted access, therefor outside of this repo and not linked


def download_dataset():
    """Download dataset from azure, ensure that config file form azure exists."""
    import convert_azure_to_tfrecords

    """Download dataset from azure, 'LorenzTao' repository required"""
    parser = ArgumentParser("Convert an Azure Machine Learning Labeled dataset to tf records.")
    parser.add_argument("-dataset_name", type=str, default="port_ai_20220216_100715")
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument(
        "-d",
        "--download",
        action="store_true",
        default=False,
        help="Download data from dataset (uneccesary if data already exists on disk)",
    )
    parser.add_argument(
        "-l",
        "--local",
        action="store_true",
        default=True,
        help="Add if you run locally instead of through an experiment in azure.",
    )
    parser.add_argument("-t", "--type", choices=["kitti", "tfrecord"], default="kitti")
    args = parser.parse_args()
    if os.path.exists(args.output_path):
        shutil.rmtree(args.output_path)
    convert_azure_to_tfrecords.main(args)
    # if not os.path.isdir(args.output_path):
    # else:
    # print("Skipped download, dataset already exist in {0}".format(args.output_path))


def prepare_dataset(
    croppedOrFullImages, dataset_class, discard_bad_detections, centered_percentage, min_bounding_box_size
):
    """Prepare dataset (kitti_format, remove discard data, split samples, create train/test dir"""
    root_dir = os.path.join(file_path, "../../output_{0}".format(centered_percentage))
    # separate_broken_samples.remove_bad_data(root_dir, p.join(file_path, "../data/ignore_samples.txt"))

    kitti_format_handler.kitti_to_imagefolder_format(root_dir, discard_bad_detections, centered_percentage)

    separate_broken_samples.remove_bad_data(
        os.path.join(root_dir, croppedOrFullImages, dataset_class),
        p.join(file_path, "../data/{0}/discard_samples.txt".format(dataset_class)),
    )
    separate_broken_samples.split_samples(
        os.path.join(root_dir, croppedOrFullImages, dataset_class),
        os.path.join(file_path, "../data/{0}/broken_samples.txt".format(dataset_class)),
    )


if __name__ == "__main__":
    centered_percentage = 1
    
    dirs=["/media/jan/Data/ubuntu_data_dir/git/MasterThesis/deepstream/out_crops_train_val/out_crops_train/stream_0/Ladder",
        "/media/jan/Data/ubuntu_data_dir/git/MasterThesis/deepstream/out_crops_train_val/out_crops_val/stream_0/Ladder",
    ]
    types=["../../../../ladder/multiclass/train","../../../../ladder/multiclass/test"]

    subdir=""
    for i in range(len(dirs)):
        # kitti_format_handler.kitti_to_imagefolder_format(dir, True, centered_percentage)
        separate_broken_samples.split_multiclass_samples(
        dirs[i],
        ["/media/jan/Data/ubuntu_data_dir/git/MasterThesis/data/ladder/multiclass/bended_ladder_samples_2022.txt",
        "/media/jan/Data/ubuntu_data_dir/git/MasterThesis/data/ladder/multiclass/blocked_ladder_samples_2022.txt",
        "/media/jan/Data/ubuntu_data_dir/git/MasterThesis/data/ladder/multiclass/rope_on_ladder_samples_2022.txt"],
        "/media/jan/Data/ubuntu_data_dir/git/MasterThesis/data/ladder/multiclass/discard_samples_2022.txt", types[i]
    )