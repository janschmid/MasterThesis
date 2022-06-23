"""This file should contain all application specific datasets"""
import os

from PIL import Image
from torch.utils.data import Dataset


class FenderDataset(Dataset):
    """Custom dataset for images with labels based on folder structure, example:
    MyFolder/label1/image1.jpg
    MyFolder/label1/image2.jpg
    MyFolder/label2/image4.jpg
    MyFolder/label2/image5.jpg
    """

    def __init__(self, root_path, transform=None, target_transform=None, shuffel=True):
        super().__init__()
        image_paths, image_classes, labels = self._get_subfolder(root_path)
        self.targets = image_classes
        self.img_paths = image_paths
        self.transform = transform
        self.target_transform = target_transform
        self.shuffel = shuffel
        self.labels = labels
        

    @staticmethod
    def _get_subfolder(root_path):
        """Get all the path to the images and save them in a list
        image_paths and the corresponding label in image_paths"""
        training_names = os.listdir(root_path)
        if(len(training_names)==2 and "broken" in training_names and "unbroken" in training_names):
            training_names=["unbroken", "broken"]
        image_paths = []
        image_classes = []
        class_id = 0
        for training_name in training_names:
            training_dir = os.path.join(root_path, training_name)
            class_path = FenderDataset.imlist(training_dir)
            image_paths += class_path
            image_classes += [class_id] * len(class_path)
            class_id += 1

        return image_paths, image_classes, training_names

    def __len__(self):
        return len(self.targets)

    @staticmethod
    def pil_loader(path):
        """Open image, required to prevent tensorflow error"""
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def __getitem__(self, index):
        """overwrite method from Dataset"""
        image = FenderDataset.pil_loader(self.img_paths[index])
        label = self.targets[index]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, self.img_paths[index]

    @staticmethod
    def imlist(path):
        """
        The function imlist returns all the names of the files in
        the directory path supplied as argument to the function.
        """
        return [
            os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
            # and "copy" not in f
        ]
        # return [os.path.join(path, f) for f in (os.listdir(path) if os.path.isfile(f))]
