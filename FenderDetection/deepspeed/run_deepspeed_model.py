"""Main function for model.py, runs model.py, use custom dataloader"""
from __future__ import division, print_function

import configparser
import os
from cv2 import transform

import torch
import torchvision
from torchvision import transforms
from argparse import ArgumentParser
import kornia as K
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)
script_dir = os.path.dirname(os.path.realpath(__file__))
import shutil
import __init__
from custom_dataset import FenderDataset
from model import DeepspeedModel




def load_data(active_config, batch_size, test_set_only=False):
    """Load FenderDataset data with dataloader
    :param root_dir: root dir of dataset
    :param input_size: resize size of iamges
    :param batch_size: batch_size, used for training/inference
    :param normalize: if True, images are normalized to: [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    :return dataloader"""
    data_transforms = model.get_data_transform(active_config)
    root_dir = active_config.get("dataloader_path")
    # Create training and validation datasets
    if(test_set_only):
        if("inference_path" in active_config):
            image_datasets = {x: FenderDataset(active_config.get("inference_path"), data_transforms[x]) for x in ["test"]}
        else:
            image_datasets = {x: FenderDataset(root_dir + x, data_transforms[x]) for x in ["test"]}
        dataloaders_dict = {
            x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
            for x in ["test"]
        }
    else:
        image_datasets = {x: FenderDataset(root_dir + x, data_transforms[x]) for x in ["train", "test"]}
    # Create training and validation dataloaders
        dataloaders_dict = {
            x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
            for x in ["train", "test"]
        }
    # if active_config.getboolean("grayscale"):
    #     pilTransform = torchvision.transforms.ToPILImage("L")
    pilTransform = torchvision.transforms.ToPILImage("RGB")
    # if not test_set_only:
    pilTransform(image_datasets["test"][0][0]).convert("RGB").save("TestZero.png")
    # x_gray = K.color.rgb_to_grayscale(image_datasets["test"][0][0])
    # x_laplacian: torch.Tensor = K.filters.canny(x_gray)[0]
    
    return dataloaders_dict


def copy_model_to_deepstream_config(config, model_file_path):
    """Copy model to deepstream pytorch config model dir, remove .engine to force regeneration after export"""
    try:
        target_dir = os.path.join(
            script_dir, "../../deepstream/config", config.get("copy_to_deepspeed_config_after_export")
        )
        shutil.copyfile(model_file_path, target_dir + ".onnx")
        os.remove(target_dir + ".onnx_b1_gpu0_fp32.engine")
        shutil.copyfile(model_file_path, target_dir + ".onnx")
    except:
        print("Copy model to deepstream config failed...")


if __name__ == "__main__":
    parser = ArgumentParser("Run deepspeed model.")
    parser.add_argument("--iniFileName", default="resnet.ini")
    args = parser.parse_args()

    config_file = os.path.join("FenderDetection/deepspeed/", args.iniFileName)
    config = configparser.ConfigParser()
    config.read_string(open(config_file, "rt").read())
    active_config = config[config.get("general", "active_config")]
    # if running inference, we need about double the gpu ram, so let's take only half the batch size

    if "train" in config.get("general", "execution_type"):
        model = DeepspeedModel(active_config)
        dataloader = load_data(active_config, active_config.getint("batch_size"))
        model.train(dataloader)
        with open(config_file, "w") as configfile:
            config.write(configfile)
        export_config =  configparser.ConfigParser()
        export_config["DEFAULT"]=active_config
        with open(os.path.join(active_config.get("trained_model_path"), "config.txt"), "w") as active_config_export:
            export_config.write(active_config_export)

    if "inference" in config.get("general", "execution_type"):
        print("Changing active config, load config from saved model")
        model = DeepspeedModel.init_by_config(active_config.get("trained_model_path"))
        dataloader = load_data(model.m_config, int(model.m_config.getint("batch_size")/2), test_set_only=False)
        embeddings = None
        # embeddings = model.get_embeddings(dataloader["test"])
        result_table = model.run_inference(dataloader["test"])
        result_table = model.calculate_cam_for_result_table(result_table)
        model.visualize(result_table, embeddings)
    if "export" in config.get("general", "execution_type"):
        model = DeepspeedModel.init_by_config(active_config.get("trained_model_path"))
        onnx_file_path = model.export_to_onnx(active_config.get("trained_model_path"))
        copy_model_to_deepstream_config(active_config, onnx_file_path)


