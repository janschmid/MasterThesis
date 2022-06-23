"""This file handles the kitti dataformat https://docs.nvidia.com/tao/tao-toolkit/text/data_annotation_format.html?highlight=kitti"""

import os

import imagesize  # get size without opening

file_path = os.path.dirname(os.path.realpath(__file__))
import random
import re
import shutil
from pathlib import Path

import cv2 as cv2
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
import __init__
from lorenz_dataset import Label
from lorenz_dataset import LorenzDataset
from PIL import ImageColor

colors = sns.color_palette("Paired", 9 * 2).as_hex()
names = ["safety_ladder", "fender", "ladder"]
fender_min_detection_size = (100, 50)


def get_labels_from_kitti(label_file, get_image_size=False):
    """parse labels from kitti label file
    :param label_file: path to label file
    :param get_image_size: return height and width
    :return labels[] if get_image_size==False
    :return labels[],width,height if get_image_size==True"""
    with open(label_file, "r") as f:
        df = pd.read_csv(
            label_file,
            delimiter=" ",
            header=None,
            names=[
                "class_name",
                "truncation",
                "olcclusion",
                "alpha",
                "bbox_xmin",
                "bbox_ymin",
                "bbox_xmax",
                "bbox_ymax",
                "3d_height",
                "3d_width",
                "3d_length",
                "location_x",
                "location_y",
                "location_z",
                "rotation_y",
            ],
        )
        d = dict(tuple(df.groupby("class_name")))
        labels = []
        for df_class in df["class_name"].unique():
            if len(df_class) > 1:
                count = 1
            else:
                count = 0
            for entry in d[df_class].itertuples():
                label = Label(
                    entry.bbox_xmin,
                    entry.bbox_xmax,
                    entry.bbox_ymin,
                    entry.bbox_ymax,
                    entry.class_name,
                    "{0}{1}".format(
                        os.path.basename(label_file).split(".")[0],
                        "" if count == 0 else "_detection_{0}".format(count),
                    ),
                )
                labels.append(label)
                count += 1

        if get_image_size:
            path_to_img = os.path.join(Path(label_file).parents[1], "images", Path(label_file).stem + ".jpg")
            width, height = imagesize.get(path_to_img)
            return labels, width, height
        return labels


def check_if_detection_is_in_center(img_width, centered_percentage, bbox_left, bbox_right):
    """Check location of the bounding box, if center_percentage is 100, return true wherever bounding box is.
    If center_percentage is 50, return true if bounding box is outisde of 1st and 3rd quarter of image. -> left and right 0.25 percent are 'outside'
    :param image_width: width of image
    :param centered_percentage boudning box thershold [0-100], 0: nowhere, 100: everywhere, 50: inner 50%
    :param bbox left: left corner of bounding box
    :param bbox_right: right corner of bounding box"""
    # Check if detectin is in the center or cornder
    left_limit_percentage = (1 - centered_percentage) / 2
    right_limit_percentage = 1 - left_limit_percentage
    if bbox_left > (img_width * left_limit_percentage) and bbox_right < (img_width * right_limit_percentage):
        return True
    else:
        return False

def check_if_detection_is_on_edge(img_width, bbox_left, bbox_right):
    """Check if bounding box starts/ends at image start/end, 5 pixel buffer by default"""
    # Check if detectin is in the center or cornder
    if bbox_left <= 5 or bbox_right+5>=img_width:
        return False
    else:
        return True

def check_min_bounding_box_size(bbox_left, bbox_right, min_size=50):
    """Simple check width of bounding box is at least min_size"""
    if bbox_right - bbox_left < min_size:
        return False
    else:
        return True


def _kitti_to_imagefolder_format_parallel(
    root_dir, label_file, centered_percentage, discard_bad_detections=True, min_bounding_box_size=70
):
    """Call kitti_to_imagefolder_format instead, which calls this one in parallel"""
    label_dir = os.path.join(root_dir, "labels")
    image_dir = os.path.join(root_dir, "images")
    labels = []
    label_file = os.path.join(label_dir, label_file)
    for label in get_labels_from_kitti(label_file):
        if discard_bad_detections:
            labels.append(label)
            imgPath = os.path.join(image_dir, os.path.basename(label_file).replace(".txt", ".jpg"))
            img = cv2.imread(imgPath)
            if img is None:
                print("Could not find image: {0}, continue".format(imgPath))
            _, img_width, _ = img.shape

            if not check_if_detection_is_in_center(
                img_width, centered_percentage, label.xmin, label.xmax
            ) or not check_min_bounding_box_size(label.xmin, label.xmax, min_bounding_box_size)  or not check_if_detection_is_on_edge(img_width, label.xmin, label.xmax):
                # not in center of image
                if label.name == "ladder":
                    continue
                draw_bounding_box(img, label)
                cv2.imwrite(
                    os.path.join(
                        root_dir,
                        "sorted_out",
                        str(check_min_bounding_box_size(label.xmin, label.xmax))
                        + "_"
                        + str(+random.randint(0, 1000))
                        + os.path.basename(label_file).replace(".txt", ".jpg"),
                    ),
                    img,
                )
                continue
        copy_cropped_image_to_imagefolder(root_dir, label)
    copy_full_image_to_imagefolder(root_dir, labels)


def kitti_to_imagefolder_format(root_dir, discard_bad_detections=True, centered_percentage=1):
    """folder has to be structured in kitti format, in root dir the folders `labels` and `images` have to exist,
    copy cropped and full images to `cropped_images` and `full_images` folder,
    cropped imags get counted by index of label,
    draw bounding boxes in full image"""
    label_dir = os.path.join(root_dir, "labels")
    # for label_file in sorted(os.listdir(label_dir)):
    Parallel(n_jobs=6)(
        delayed(_kitti_to_imagefolder_format_parallel)(
            root_dir, label_file, centered_percentage, discard_bad_detections
        )
        for label_file in sorted(os.listdir(label_dir))
    )


def draw_bounding_box(img, label):
    """Draw bounding for label on image, since image is an array, the image will be changed,
    create a copy/deepcopy before this function call if original image is required for further usage"""
    xmin = int(float(label.xmin))
    xmax = int(float(label.xmax))
    ymin = int(float(label.ymin))
    ymax = int(float(label.ymax))
    color = ImageColor.getcolor(colors[names.index(label.name)], "RGB")
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness=2)
    cv2.putText(img, label.name, (xmin, ymin - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, color)


def copy_full_image_to_imagefolder(root_dir, labels):
    """Copy image to imagefolder and draw bounding boxes, combination of kitti and imagefolder https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html"""
    if len(labels) == 0:
        return
    labels_path = os.path.join(root_dir, "full_images", "labels")
    images_path = os.path.join(root_dir, "full_images", "images")
    images_path_bbox = os.path.join(root_dir, "full_images_bbox", "images")
    os.makedirs(labels_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)
    source_label_name = re.sub(r"_detection_[0-9]*", r"", labels[0].text)
    shutil.copyfile(
        os.path.join(root_dir, "labels", source_label_name + ".txt"),
        os.path.join(labels_path, source_label_name + ".txt"),
    )

    img = cv2.imread(os.path.join(root_dir, "images", source_label_name + ".jpg"))
    cv2.imwrite(os.path.join(images_path, source_label_name + ".jpg"), img)
    for label in labels:
        draw_bounding_box(img, label)
    cv2.imwrite(os.path.join(images_path_bbox, source_label_name + ".jpg"), img)


def copy_cropped_image_to_imagefolder(root_dir, label):
    """Crop image and save cropped image in imagefolder https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html"""
    output_base_path = os.path.join(root_dir, "cropped_images", label.name, "images")
    os.makedirs(output_base_path, exist_ok=True)
    source_label_name = re.sub(r"_detection_[0-9]*", r"", label.text)
    image_path = os.path.join(root_dir, "images", source_label_name + ".jpg")
    target_path = os.path.join(output_base_path, label.text + ".jpg")
    img = cv2.imread(image_path)
    xmin = int(float(label.xmin))
    xmax = int(float(label.xmax))
    ymin = int(float(label.ymin))
    ymax = int(float(label.ymax))
    cropped_img = img[ymin:ymax, xmin:xmax]
    cv2.imwrite(target_path, cropped_img)

def draw_bounding_box_on_single_image(image_path, label_path):
    img = cv2.imread(image_path)
    for label in get_labels_from_kitti(label_path):
        draw_bounding_box(img, label)
    return img

def discard_small_fenders():
    labels_dir = "/media/jan/Data/ubuntu_data_dir/git/MasterThesis/HelperFunctions/test_labels"
    for file in os.listdir(labels_dir):
        path = os.path.join(labels_dir, file)
        labels = get_labels_from_kitti(path)
        non_discarded_labels = []
        for label in labels:
            if(label.name.lower() != "fender"):
                non_discarded_labels.append(label)
            elif(label.ymax-label.ymin>120):
                non_discarded_labels.append(label)
        kitti=[]
        for label in non_discarded_labels:
            kitti.append(LorenzDataset.bbox_to_kitti(label))
        kitti_label_text = '\n'.join(kitti)
        with open(path, 'w') as file:
            file.write(kitti_label_text)
            



if __name__ == "__main__":
    # root_dir = os.path.join(file_path, "../../output")
    # kitti_to_imagefolder_format(root_dir)
    # _kitti_to_imagefolder_format_parallel(root_dir, "portai20211215111931_IMG_0011_copy_26.txt")
    discard_small_fenders()
    # img = draw_bounding_box_on_single_image("/media/jan/Data/ubuntu_data_dir/git/output_0.8/full_images/images/portai20211215111931_IMG_0011_copy_9.jpg",
    #     "/media/jan/Data/ubuntu_data_dir/git/output_0.8/labels/portai20211215111931_IMG_0011_copy_9.txt")
    # cv2.imwrite("/media/jan/Data/ubuntu_data_dir/git/MasterThesis/ztest1.png", img)
