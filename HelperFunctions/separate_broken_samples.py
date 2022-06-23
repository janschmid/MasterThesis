"""Update list of broken samples since they are not labeled in azureml/labelbox so far.
Idea: move all images manually to a folder and create a csv list out of the broken images"""
from genericpath import exists
import os
import os.path as p
import re
import shutil
from argparse import ArgumentParser
from os import listdir, makedirs, mkdir

import pandas as pd
from sklearn.model_selection import train_test_split

script_dir = p.dirname(p.realpath(__file__))
data_dir = p.join(script_dir, "../data")


class BrokenSamplesDetector:
    """Check if image is in borken table or not, class required to load broken samples text file only at instanciation of class"""

    _broken_samples_table = []
    _unbroken_index=None
    def __init__(self, lables_file_path, label_type="fender", is_2022=False) -> None:
        with open(lables_file_path) as f:
            lines=f.readlines()
        for line in lines:
            line = line.replace("\n","")
            if not ("unbroken" in line):
                path = p.join(data_dir, label_type if(len(lines)==2) else label_type+"/multiclass", line+"_samples_2022.txt" if is_2022 else "_samples.txt")
                self._broken_samples_table.append(pd.read_csv(path, header=None).to_numpy())
            else:
                self._broken_samples_table.append(None)
                self._unbroken_index=len(self._broken_samples_table)-1
        if(self._unbroken_index==None):
            raise ValueError("Couldn't find unborken index, initialize borken samples detector failed")


    def is_broken(self, file_name):
        """Returns ture if the sample is broken"""
        for i in range(len(self._broken_samples_table)):
            if file_name in self._broken_samples_table[i]:
                return i
            else:
                return self._unbroken_index


def remove_bad_data(root_dir, bad_sampels_file):
    """Ignore wrong labeled data by removing them from the root folder,
    Removes label if exist, if not ignored"""
    bad_samples = pd.read_csv(bad_sampels_file, header=None).to_numpy()
    images_dir = p.join(root_dir, "images")
    labels_dir = p.join(root_dir, "labels")
    for file in os.listdir(images_dir):
        if file in bad_samples:
            bad_image_path = p.join(images_dir, file)
            bad_label_path = p.join(labels_dir, file)
            os.remove(bad_image_path)
            label_path = bad_label_path.replace(".jpg", ".txt")
            if p.exists(label_path):
                os.remove(label_path)
            else:
                print("Label not found, therefore deletion ignored: {0}".format(label_path))


# def validate_and_split_inference_data(manual_cropped_images_dir, inference_images_dir):
#     """"""
#     manual_cropped_images = os.listdir(manual_cropped_images_dir)
#     correct_classified_inference_dir = p.join(inference_images_dir, "../", "correct_classified")
#     incorrect_classified_inference_dir = p.join(inference_images_dir, "../", "incorrect_classified")
#     os.makedirs(correct_classified_inference_dir, exist_ok=True)
#     os.makedirs(incorrect_classified_inference_dir, exist_ok=True)
#     for file in os.listdir(inference_images_dir):
#         if file in manual_cropped_images:
#             shutil.copy(p.join(inference_images_dir, file), p.join(correct_classified_inference_dir, file))
#         else:
#             shutil.copy(p.join(inference_images_dir, file), p.join(incorrect_classified_inference_dir, file))


def update_broken_samples_table(path_to_broken_samples):
    """Update table based on the files in the broken folder directory"""
    broken_samples = []
    for file in listdir(path_to_broken_samples):
        broken_samples.append(p.basename(file))
        broken_samples.append(re.sub(r"_detection_[0-9]*", r"", file))
    return broken_samples


def append_broken_samples(result_table_path, samples_to_add):
    """Append files to existing list if not exist, remove duplicates"""
    samples_to_add = pd.DataFrame(samples_to_add)
    if p.exists(result_table_path):
        broken_samples = pd.read_csv(result_table_path, header=None)
        samples_to_add = pd.concat([broken_samples, samples_to_add]).drop_duplicates().reset_index(drop=True)
    samples_to_add.columns = ["name"]
    samples_to_add = samples_to_add.sort_values(by="name")
    samples_to_add.to_csv(result_table_path, index=False, header=False)


def split_samples(input_base_dir, broken_samples_file, split_train_eval=True):
    """Split samples based on path to broken samples file to broken and unbroken"""
    broken_samples = pd.read_csv(broken_samples_file, header=None).to_numpy()
    num_broken = 0
    num_unbroken = 0
    unbroken_img_dir = p.join(input_base_dir, "images_unbroken")
    broken_img_dir = p.join(input_base_dir, "images_broken")
    makedirs(broken_img_dir, exist_ok=True)
    makedirs(unbroken_img_dir, exist_ok=True)

    for file in sorted(listdir(p.join(input_base_dir, "images"))):
        if file in broken_samples:
            shutil.copyfile(p.join(input_base_dir, "images", file), p.join(broken_img_dir, file))
            num_broken += 1
        else:
            shutil.copyfile(p.join(input_base_dir, "images", file), p.join(unbroken_img_dir, file))
            num_unbroken += 1
    print(
        "Total: {0}, Broken: {1}, Known broken {2}, Unbroken: {3}\n saved to: {4}".format(
            num_broken + num_unbroken, num_broken, len(broken_samples), num_unbroken, broken_img_dir
        )
    )
    if split_train_eval:
        broken_train_dir, broken_test_dir = split_train_test(broken_img_dir)
        unbroken_train_dir, unbroken_test_dir = split_train_test(unbroken_img_dir)
        create_fender_dataset_symlink(input_base_dir, broken_train_dir, unbroken_train_dir, "train")
        create_fender_dataset_symlink(input_base_dir, broken_test_dir, unbroken_test_dir, "test")

def split_multiclass_samples(input_base_dir, broken_samples_files, discard_samples_file, split_train_eval=True):
    num_broken = 0
    num_unbroken = 0
    input_base_dir = p.join(input_base_dir, "cropped_images_multiclass")
    unbroken_img_dir = p.join(input_base_dir, "images_unbroken")
    shutil.copytree(p.join(input_base_dir, "images"), unbroken_img_dir, dirs_exist_ok=True)
    discard_samples =  pd.read_csv(discard_samples_file, header=None).to_numpy()
    for file in os.listdir(unbroken_img_dir):
        if(file in discard_samples):
            os.remove(file)
    # broken_samples = pd.read_csv(broken_samples_file, header=None).to_numpy()
    for broken_samples_file in broken_samples_files:
        broken_samples_name = os.path.basename(broken_samples_file).split("_samples_")[0]
        broken_samples = pd.read_csv(broken_samples_file, header=None).to_numpy()
        broken_img_dir = p.join(input_base_dir, broken_samples_name)
        makedirs(broken_img_dir, exist_ok=True)
        
        for file in sorted(listdir(unbroken_img_dir)):
            if file in broken_samples:
                shutil.move(p.join(unbroken_img_dir, file), p.join(broken_img_dir, file))
                num_broken += 1
        
        if(split_train_eval==True):
            broken_train_dir, broken_test_dir = split_train_test(broken_img_dir)
            # create_fender_dataset_symlink(input_base_dir, broken_train_dir, unbroken_train_dir, "train")
        # print(
        #     "Total: {0}, Broken: {1}, Known broken {2}, Unbroken: {3}\n saved to: {4}".format(
        #         num_broken + num_unbroken, num_broken, len(broken_samples), num_unbroken, broken_img_dir
        #     )
        # )
        if split_train_eval==True:
            train_dir = p.join(input_base_dir, "train")
            test_dir = p.join(input_base_dir, "test")
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)
            if not p.exists(p.join(train_dir, broken_samples_name)):
                os.symlink(broken_train_dir, p.join(train_dir, broken_samples_name))
            if not p.exists(p.join(test_dir, broken_samples_name)):
                os.symlink(broken_test_dir, p.join(test_dir, broken_samples_name))
        else:
            if(split_train_eval is not None and split_train_eval is not False):
                split_dir = p.join(input_base_dir, split_train_eval)
                os.makedirs(split_dir, exist_ok=True)
                if not p.exists(p.join(split_dir, broken_samples_name)):
                    os.symlink(broken_img_dir, p.join(split_dir, broken_samples_name))

    
    if split_train_eval ==True :
        train_dir = p.join(input_base_dir, "train")
        test_dir = p.join(input_base_dir, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        broken_samples_name=p.basename(unbroken_img_dir)
        if not p.exists(p.join(train_dir, broken_samples_name)):
            os.symlink(unbroken_img_dir, p.join(train_dir, broken_samples_name))
        if not p.exists(p.join(test_dir, broken_samples_name)):
            os.symlink(unbroken_img_dir, p.join(test_dir, broken_samples_name))
    else:
        if(split_train_eval is not None and split_train_eval is not False):
            split_dir = p.join(input_base_dir, split_train_eval)
            os.makedirs(split_dir, exist_ok=True)
            broken_samples_name=p.basename(unbroken_img_dir)
            if not p.exists(p.join(split_dir, broken_samples_name)):
                os.symlink(unbroken_img_dir, p.join(split_dir, broken_samples_name))
            # os.makedirs(full_train_test_dir, exist_ok=True)
            # broken_train_dir, broken_test_dir = split_train_test(broken_img_dir)
            # unbroken_train_dir, unbroken_test_dir = split_train_test(unbroken_img_dir)
            # create_fender_dataset_symlink(input_base_dir, broken_train_dir, unbroken_train_dir, "train")
            # create_fender_dataset_symlink(input_base_dir, broken_test_dir, unbroken_test_dir, "test")


def create_fender_dataset_symlink(base_dir, broken_dir, unbroken_dir, train_test_dir_name):
    """Create symlink to separate samples to image dataset with "train" and "test" folder"""
    full_train_test_dir = p.join(base_dir, train_test_dir_name)
    os.makedirs(full_train_test_dir, exist_ok=True)
    if not p.exists(p.join(full_train_test_dir, "broken")):
        os.symlink(broken_dir, p.join(full_train_test_dir, "broken"))
    if not p.exists(p.join(full_train_test_dir, "unbroken")):
        os.symlink(unbroken_dir, p.join(full_train_test_dir, "unbroken"))


def clean_or_create_directory(directory):
    """Function for a "fresh start" in a directory, clean all files in the directory (if there are),
    or create the directory and all intermediate ones"""
    if p.exists(directory):
        shutil.rmtree(directory)
        mkdir(directory)
    else:
        makedirs(directory)


def split_train_test(input_base_dir, create_symlink=False):
    """Split train and test data, replace whitespaces to prevent errors in furhter pipeline,
    creates now folder with _train and _test where data is inside
    :return: train_dir, test_dir"""
    f = []
    for (dirpath, dirnames, filenames) in os.walk(input_base_dir):
        f.extend(filenames)
        break

    x_train, x_eval = train_test_split(f, test_size=0.2)
    train_dir = input_base_dir + "_train"
    eval_dir = input_base_dir + "_test"
    any(map(clean_or_create_directory, [train_dir, eval_dir]))
    
    for file in x_train:
        shutil.copyfile(p.join(input_base_dir, file), p.join(train_dir, file.replace(" ", "_")))
    for file in x_eval:
        shutil.copyfile(p.join(input_base_dir, file), p.join(eval_dir, file.replace(" ", "_")))
    if(create_symlink):
        subdir_name = os.path.basename(input_base_dir)
        for name in ['train', 'test']:
            os.makedirs(os.path.join(os.path.dirname(input_base_dir), name), exist_ok=True)
            broken_symlink_dir = os.path.join(os.path.dirname(input_base_dir), name, subdir_name)
            if not p.exists(broken_symlink_dir):
                os.symlink(input_base_dir + '_' + name, broken_symlink_dir)
    return train_dir, eval_dir


if __name__ == "__main__":
    # broken_img_dir="/media/jan/Data/ubuntu_data_dir/git/MasterThesis/deepstream/out_crops_train_val/mixed/fender/broken"
    # unbroken_img_dir="/media/jan/Data/ubuntu_data_dir/git/MasterThesis/deepstream/out_crops_train_val/mixed/fender/unbroken"
    input_base_dir="/media/jan/Data/ubuntu_data_dir/git/MasterThesis/deepstream/out_crops_train_val/ladder/multiclass/randomSplit"
    for dir in os.listdir(input_base_dir):
        if ('train' not in dir and 'test' not in dir and ('ladder' in dir or 'unbroken' in dir)):
            split_train_test(os.path.join(input_base_dir, dir), True)
    # broken_train_dir, broken_test_dir = split_train_test(broken_img_dir)
    # unbroken_train_dir, unbroken_test_dir = split_train_test(unbroken_img_dir)
    # create_fender_dataset_symlink(input_base_dir, broken_train_dir, unbroken_train_dir, "train")
    # create_fender_dataset_symlink(input_base_dir, broken_test_dir, unbroken_test_dir, "test")
    # parser = ArgumentParser("Update broken samples based on directory.")
    # parser.add_argument(
    #     "-b",
    #     "--broken_samples_dir",
    #     type=str,
    #     help="Root directory where broken samples are lying",
    #     default="/home/jan-ruben/Git/output/cropped_images/fender/images_broken",
    # )
    # parser.add_argument(
    #     "-r",
    #     "--result_table_path",
    #     type=str,
    #     help="Numpy txt array were broken file names are stored, if exist appended",
    #     default="broken_samples.txt",
    # )
    # args = parser.parse_args()

    # broken_fenders = update_broken_samples_table(args.broken_samples_dir)
    # append_broken_samples(args.result_table_path, broken_fenders)
