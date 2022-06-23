"""Train pytorch with deepspeed engine"""
import copy
import datetime
import json
import os
import time
import __init__
import common as cm
import deepspeed
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from classification_report import Report
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorchtools import EarlyStopping
from separate_broken_samples import BrokenSamplesDetector
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score
from torch2trt import torch2trt
from torchvision import models
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
from torchvision import transforms
import torchvision
import configparser
import pprint

class CannyEdge(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        edge = K.filters.canny(img.unsqueeze(0))[0]
        return edge.squeeze(0).repeat(3,1,1)

class DeepspeedModel:
    """Main idea is to have an wrapper with an ini file for pytorch and deepspeed.
    The pytorch model is used and sent to deepspeed before each run"""

    _m_model = None
    _m_script_dir = os.path.dirname(os.path.realpath(__file__))
    m_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    m_config = None

    def __init__(self, model_config):
        """Initialize the model with a config, a detailed docu of the config does not exist, but it should be self explaining,
        see the passed .ini file. Pass only the 'model' section
        :param model_config: 'model' section of ini file
        :return: no return values, does set self params"""
        self.m_config = model_config
        self._m_run_name = "{1}_{0}".format(self.m_config.name, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        self.__initialize_model(
            model_config.get("name"),
            len(eval(model_config.get("class_names"))),
            model_config.get("num_layers"),
            model_config.getboolean("use_pretrained", fallback=True),
        )

    @staticmethod
    def init_by_config(saved_model_path):
        config = configparser.ConfigParser()
        config.read(os.path.join(saved_model_path, "config.txt"))
        model = DeepspeedModel(config['DEFAULT'])
        model.load_trained_state_dict(saved_model_path)
        return model

    def load_trained_state_dict(self, state_dict_path):
        """Load trained model, model has to be named 'pytroch_model.bin'
        :param state_dict_path: path to state dict which contains 'pytorch_model.bin'"""
        config = configparser.ConfigParser()
        state_dict = torch.load(os.path.join(state_dict_path, "pytorch_model.bin"))
        self._m_model.load_state_dict(state_dict)
        config.read(os.path.join(state_dict_path, "config.txt"))
        self.m_config=config['DEFAULT']

    def export_to_tensorrt(self, output_folder):
        """Export model to tensorrt file, export config must be same as train config"""
        input_size = eval(self.m_config.get("input_size"))
        x = torch.ones((1, 3, input_size[0], input_size[1])).to(self.m_device)
        model_trt = torch2trt(self._m_model, [x])
        with open(os.path.join(output_folder, "deepstream_model.engine"), "wb") as f:
            f.write(model_trt.engine.serialize())

    def export_to_onnx(self, output_folder):
        """Export model to onnx file, export config must be same as train config"""
        input_size = eval(self.m_config.get("input_size"))
        dummy_input = torch.ones((1, 3, input_size[0], input_size[1])).to(self.m_device)
        input_names = ["data"]
        output_names = ["output"]
        onnx_file_path = os.path.join(output_folder, "deepstream_model.onnx")
        torch.onnx.export(
            self._m_model,
            dummy_input,
            onnx_file_path,
            input_names=input_names,
            opset_version=14,
            output_names=output_names,
            dynamic_axes={"data": {0: "batch"}, "output": {0: "batch"}},
            export_params=True,
        )
        return onnx_file_path

    def set_parameter_requires_grad(self):
        """Set param.requires_grad=False if feature_extracted is set to true in global config"""
        if self.m_config.getboolean("feature_extract"):
            for param in self._m_model.parameters():
                param.requires_grad = False

    def _set_last_layer_for_embeddings(self, model):
        """Replace last layer of model with fully connected to get the embeddings
        :param model: Model which has to be configured
        :return: configured model"""
        if self.m_config.get("name") == "resnet":
            num_ftrs = model.fc.in_features
            model.fc = nn.Identity(num_ftrs, num_ftrs)
            return model
        elif self.m_config.get("name") == "mobilenet_v3_small":
            num_ftrs = model.classifier[3].in_features
            model.classifier[3]=nn.Identity(num_ftrs, 1000)
            return model
        else:
            raise NotImplementedError(
                "Model {0} is not implemented, implement similar to initialize_model".format(self.m_config.get("name"))
            )
    def replace_hardswish_to_relu(model):
        for child_name, child in model.named_children():
            if isinstance(child, nn.Hardswish):
                setattr(model, child_name, nn.ReLU())
            else:
                DeepspeedModel.replace_hardswish_to_relu(child)

    def __initialize_model(self, model_name, num_classes, num_layers, use_pretrained):
        """Initialize model by name
        :param model_name: name of model, has to be ["resnet", "alexnet", "vgg", "squeeezenet", "densenet", "inception", efficientnet"]
        :param num_classes: number of output labels
        :param num_layers: resnet: number of layers [18,34,50,101,152], efficientnet: [b0-b5]
        :param use_pretrained: use pre-trained network from pytorch"""
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        self._m_model = None
        if model_name == "resnet":
            num_layers = int(num_layers)
            if num_layers == 18:
                self._m_model = models.resnet18(pretrained=use_pretrained)
            elif num_layers == 34:
                self._m_model = models.resnet34(pretrained=use_pretrained)
            elif num_layers == 50:
                self._m_model = models.resnet50(pretrained=use_pretrained)
            elif num_layers == 101:
                self._m_model = models.resnet101(pretrained=use_pretrained)
            elif num_layers == 152:
                self._m_model = models.resnet152(pretrained=use_pretrained)
            self.set_parameter_requires_grad()
            num_ftrs = self._m_model.fc.in_features
            self._m_model.fc = nn.Linear(num_ftrs, num_classes)

        elif model_name == "alexnet":
            """Alexnet"""
            self._m_model = models.alexnet(pretrained=use_pretrained)
            self.set_parameter_requires_grad()
            num_ftrs = self._m_model.classifier[6].in_features
            self._m_model.classifier[6] = nn.Linear(num_ftrs, num_classes)

        elif model_name == "vgg":
            """VGG11_bn"""
            self._m_model = models.vgg11_bn(pretrained=use_pretrained)
            self.set_parameter_requires_grad()
            num_ftrs = self._m_model.classifier[6].in_features
            self._m_model.classifier[6] = nn.Linear(num_ftrs, num_classes)

        elif model_name == "squeezenet":
            """Squeezenet"""
            self._m_model = models.squeezenet1_0(pretrained=use_pretrained)
            self.set_parameter_requires_grad()
            self._m_model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
            self._m_model.num_classes = num_classes

        elif model_name == "densenet":
            """Densenet"""
            self._m_model = models.densenet121(pretrained=use_pretrained)
            self.set_parameter_requires_grad()
            num_ftrs = self._m_model.classifier.in_features
            self._m_model.classifier = nn.Linear(num_ftrs, num_classes)

        elif model_name == "inception":
            """Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            self._m_model = models.inception_v3(pretrained=use_pretrained)
            self.set_parameter_requires_grad()
            # Handle the auxilary net
            num_ftrs = self._m_model.AuxLogits.fc.in_features
            self._m_model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = self._m_model.fc.in_features
            self._m_model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name == "efficientnet_b0":
            self._m_model = models.efficientnet_b0(pretrained=use_pretrained)
            self.set_parameter_requires_grad()
            num_ftrs = self._m_model.classifier[1].in_features
            self._m_model.classifier[1]=nn.Linear(num_ftrs, num_classes)
        elif model_name =="convnext_tiny":
            self._m_model = models.convnext_tiny(pretrained=use_pretrained)
            self.set_parameter_requires_grad()
            num_ftrs = self._m_model.classifier[2].in_features
            self._m_model.classifier[2]=nn.Linear(num_ftrs, num_classes)
        elif model_name == "mobilenet_v3_small":
            self._m_model = models.mobilenet_v3_small(pretrained=use_pretrained, use_hs=False)
            DeepspeedModel.replace_hardswish_to_relu(self._m_model)
            self.set_parameter_requires_grad()
            num_ftrs = self._m_model.classifier[3].in_features
            self._m_model.classifier[3]=nn.Linear(num_ftrs, num_classes)
        elif model_name == "mobilenet_v3_large":
            self._m_model = models.mobilenet_v3_large(pretrained=use_pretrained)
            self.set_parameter_requires_grad()
            num_ftrs = self._m_model.classifier[3].in_features
            self._m_model.classifier[3]=nn.Linear(num_ftrs, num_classes)
        elif model_name == "mobilenet_v2":
            self._m_model = models.mobilenet_v2(pretrained=use_pretrained)
            self.set_parameter_requires_grad()
            num_ftrs = self._m_model.classifier[1].in_features
            self._m_model.classifier[1]=nn.Linear(num_ftrs, num_classes)
        else:
            print("Invalid model name, exiting...")
            exit()

        self.m_params_to_update = self._m_model.parameters()
        if self.m_config.getboolean("feature_extract"):
            self.m_params_to_update = []
            for name, param in self._m_model.named_parameters():
                if param.requires_grad is True:
                    self.m_params_to_update.append(param)
                    print("\t", name)

        self.m_optimizer = torch.optim.Adam(self.m_params_to_update)

        self._m_model = self._m_model.to(self.m_device)
        # Model setup, let's initialize it with deepspeed
        # optimizer_ft = optim.Adam(params_to_update, lr=0.001)

    def _get_deepspeed_model(self, model=None):
        """Send self._m_model to deepspeed
        :param model: if None: use self._m_model, else use passed model
        :return: deepspeed_model, optimizer"""
        if model is None:
            model = self._m_model
        with open(
            os.path.join(self._m_script_dir, self.m_config.get("deepspeed_config_path")), "r", encoding="utf8"
        ) as f:
            ds_config = json.load(f)
            ds_config["tensorboard"]["job_name"] = self._m_run_name
            model, optimizer, _, _ = deepspeed.initialize(
                config=ds_config, model=model, optimizer=self.m_optimizer, model_parameters=self.m_params_to_update
            )
            model = model.to(self.m_device)
            return model, optimizer

    def train(self, dataloaders):
        """Train model, for training parameters the initalization config is used, model is saved to fp16 model,
        path is set in `self.m_config.trained_model_path`
        :param dataloader: Data to train with
        :return: val_accuracy_history"""
        model_engine, optimizer = self._get_deepspeed_model()
        criterion = eval(self.m_config.get("loss_function")).to(self.m_device)
        is_inception = self.m_config.get("name") == "inception"
        run_name = os.path.join("tensorboard/", self._m_run_name)
        config_item_dict = {i[0]: i[1] for i in self.m_config.parser.items(self.m_config.name)}
        # Initialize metrics for hparams (Yes, it's annoying), but thanks to https://discuss.pytorch.org/t/how-to-add-graphs-to-hparams-in-tensorboard/109349/2
        report = Report(classes=eval(self.m_config.get("class_names")), log_dir=run_name)
        data, label, _ = next(iter(dataloaders["train"]))
        report.plot_model(self._m_model, dataloaders["train"].dataset[0][0].unsqueeze(0).to(self.m_device))
        since = time.time()

        val_acc_history = []
        early_stopping = EarlyStopping(patience=self.m_config.getint("early_stopping", 5), verbose=True, path=None)
        best_model_wts = copy.deepcopy(model_engine.state_dict())
        best_acc = 0.0

        for epoch in tqdm(range(self.m_config.getint("num_epochs"))):
            print("Epoch {}/{}".format(epoch, self.m_config.getint("num_epochs") - 1))
            print("-" * 10)

            # Each epoch has a training and validation phase
            for phase in ["train", "test"]:
                if phase == "train":
                    model_engine.train()  # Set model to training mode
                else:
                    model_engine.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels, path_to_file in dataloaders[phase]:
                    inputs = inputs.to(self.m_device)
                    labels = labels.to(self.m_device)

                    # zero the parameter gradients
                    self.m_optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        if is_inception and phase == "train":
                            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                            if type(criterion) == nn.CrossEntropyLoss:
                                outputs, aux_outputs = model_engine(inputs)
                                loss1 = criterion(outputs, labels)
                                loss2 = criterion(aux_outputs, labels)
                            else:
                                outputs, aux_outputs = model_engine(inputs)
                                logits1 = torch.softmax(outputs, 1)
                                logits2 = torch.softmax(aux_outputs, 1)
                                one_hot_labels = nn.functional.one_hot(
                                    labels, num_classes=len(eval(self.m_config.get("class_names")))
                                ).float()
                                loss1 = criterion(logits1, one_hot_labels)
                                loss2 = criterion(logits2, one_hot_labels)
                            loss = loss1 + 0.4 * loss2
                        else:
                            outputs = model_engine(inputs)
                            if type(criterion) == nn.CrossEntropyLoss:
                                loss = criterion(outputs, labels)
                            else:
                                logits = torch.softmax(outputs, 1)
                                one_hot_labels = nn.functional.one_hot(
                                    labels, num_classes=len(eval(self.m_config.get("class_names")))
                                ).float()
                                loss = criterion(logits, one_hot_labels)
                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == "train":
                            model_engine.backward(loss)
                            model_engine.step()

                        report.write_a_batch(
                            loss=loss,
                            batch_size=inputs.size(0),
                            actual=labels,
                            prediction=outputs,
                            train=True if phase == "train" else False,
                        )
                        report.plot_model_data_grad(at_which_iter=len(dataloaders["train"]) / 2)

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
                print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
                # deep copy the model
                if phase == "test":
                    early_stopping(epoch_loss, self._m_model)
                    val_acc_history.append(epoch_acc)
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model_engine.state_dict())

                report.writer.add_scalar("LR/{0}".format("train"), model_engine.lr_scheduler._last_lr[0], epoch)

            report.plot_an_epoch()
            if early_stopping.early_stop:
                break

            print()
        time_elapsed = time.time() - since
        print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
        print("Best val Acc: {:4f}".format(best_acc))

        # load best model weights
        model_engine.load_state_dict(best_model_wts)
        model_save_dir = os.path.join(self._m_script_dir, "saved_models")
        run_name = "{0}_acc_{1:.3f}".format(run_name, best_acc)
        save_model_path = os.path.join(model_save_dir, os.path.basename(run_name))
        os.makedirs(model_save_dir, exist_ok=True)
        model_engine.save_fp16_model(save_model_path)
        label_file = open("{0}/labels.txt".format(save_model_path), 'w')
        for label in dataloaders['train'].dataset.labels:
            label_file.write("{}\n".format(label))
        label_file.close()
        config_item_dict = {i[0]: i[1] for i in self.m_config.parser.items(self.m_config.name)}
        report.writer.add_hparams(config_item_dict, {"highest_val_acc": float(best_acc.cpu().detach())})
        self.m_config["trained_model_path"] = save_model_path
        return val_acc_history

    def run_inference(self, dataloader):
        """Run inference, loaded model is used, if `train_model` is not executed on this instance,
        use `load_trained_state_dict` to load trained model, results can be visualized with `visualize` function
        :param dataloader: inference dataloader
        :return: result_table"""
        model_engine, _ = self._get_deepspeed_model()
        model_engine.eval()
        is_2022 = True if "20220216150850"  in os.path.basename(dataloader.dataset.img_paths[0]) else False
        label_type = "fender" if "fender" in self.m_config.get("trained_model_path") else "ladder"
        detector = BrokenSamplesDetector(os.path.join(self.m_config.get("trained_model_path"), "labels.txt"), label_type, is_2022)
        results = []
        for inputs, labels, path_to_file in dataloader:
            inputs = inputs.to(self.m_device)
            labels = labels.to(self.m_device)
            outputs = model_engine(inputs)

            top_p, top_c = torch.softmax(outputs, 1).topk(1, dim=1)
            top_c = top_c.cpu().numpy()
            top_p = top_p.cpu().detach().numpy()
            labels = labels.cpu().numpy()
            for i in range(0, outputs.size()[0]):
                path = path_to_file[i]
                name = os.path.basename(path)
                results.append((labels[i], top_c[i][0], top_p[i][0], path))
        result_table = pd.DataFrame(list(results))
        result_table.columns = ["true_class", "pred_class", "pred_prob", "file_path"]
        return result_table

    def get_embeddings(self, dataloader):
        """Get embedding of model
        :param dataloader: dataloader to calculate embeddings
        :return: embeddings"""
        model = copy.deepcopy(self._m_model)
        model = self._set_last_layer_for_embeddings(model)
        model_engine, _ = self._get_deepspeed_model(model)
        model_engine.eval()
        embeddings = None
        for inputs, labels, path_to_file in dataloader:
            inputs = inputs.to(self.m_device)
            labels = labels.to(self.m_device)
            outputs = model_engine(inputs)
            outputs_np = outputs.cpu().detach().numpy()
            embeddings = np.concatenate((embeddings, outputs_np), axis=0) if embeddings is not None else outputs_np
        return embeddings

    @staticmethod
    def preprocess_image_cam(img: np.ndarray, config) -> torch.Tensor:
        """Preprocess image for grad-cam, thanks to https://github.com/jacobgil/pytorch-grad-cam"""
        # preprocessing = Compose([ToTensor(), Normalize(mean=mean, std=std)])
        preprocessing = DeepspeedModel.get_data_transform(config)['test']
        return preprocessing(img.copy()).unsqueeze(0)



    @staticmethod
    def get_data_transform(active_config):
        transform_list = []
        transform_list_train = []
        transform_list_test = []
        input_size = eval(active_config.get("input_size"))
        if active_config.getboolean("resize"):
            transform_list.append(transforms.Resize(input_size))
        if active_config.getboolean("train_horizontal_flip"):
            transform_list_train.append(transforms.RandomHorizontalFlip())
        if active_config.getboolean("test_center_crop"):
            transform_list_test.append(transforms.CenterCrop(input_size))
        transform_list.append(transforms.ToTensor())
        if active_config.getboolean("normalize"):
            transform_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        if active_config.getboolean("grayscale"):
            transform_list.append(torchvision.transforms.Grayscale(num_output_channels=3))
        if active_config.getboolean("canny_edge"):
            transform_list.append(CannyEdge())
        # transform_list.append(K.image_to_tensor)    
        # transform_list.append(K.filters.canny)
        # K.augmentation.RandomPerspective

        data_transforms = {
            "train": transforms.Compose(transform_list_train + transform_list),
            "test": transforms.Compose(transform_list_test + transform_list),
        }
        return data_transforms


    def calculate_cam(self, img_path, model_name, dry_run=False):
        """Calculate grad-cam for single image
        :param dry_run: if True, file path is returned, but is not calculated -> only useful if this function is executed 2nd time for same input
        :return: path to grad_cam image"""
        target_dir = os.path.join(os.path.dirname(img_path), "../cam")
        os.makedirs(target_dir, exist_ok=True)
        cam_name = os.path.join("cam_" + os.path.basename(img_path))
        file_path = os.path.join(target_dir, model_name, cam_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        pilTransform = torchvision.transforms.ToPILImage("RGB")
        if(os.path.exists(file_path)):
            dry_run=True
        if dry_run == False:
            if "resnet" in str(type(self._m_model)).lower():
                target_layers = [self._m_model.layer4[-1]]
            elif "mobilenet" in str(type(self._m_model)).lower():
                target_layers = [self._m_model.features[12]]
            else:
                raise NotImplementedError("Not implemented for network: {0}".format(str(type(self._m_model))))
            cam = GradCAM(model=self._m_model, target_layers=target_layers, use_cuda=True)
            cam.batch_size = self.m_config.getint("batch_size")

            rgb_img = Image.open(img_path)
            input_tensor = DeepspeedModel.preprocess_image_cam(rgb_img, self.m_config)
            rgb_img = rgb_img.resize(eval(self.m_config.get("input_size"))[::-1])
            rgb_img = np.float32(rgb_img) / 255
            targets = None
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            cam_pil_image = Image.fromarray(cam_image)
            rgb_pil_image = Image.fromarray(np.uint8(rgb_img * 255))
            if(np.allclose(np.array(rgb_pil_image), np.array(pilTransform(input_tensor.squeeze(0))), rtol=1, atol=1)):
                #input tensors and rgb image are the same, so we don't show them twice
                new_image = Image.new("RGB", (2 * input_tensor.shape[3], input_tensor.shape[2]))
                new_image.paste(pilTransform(input_tensor.squeeze(0)), (0, 0))
                new_image.paste(cam_pil_image, (input_tensor.shape[3], 0))
            else:
                new_image = Image.new("RGB", (3 * input_tensor.shape[3], input_tensor.shape[2]))
                new_image.paste(pilTransform(input_tensor.squeeze(0)), (0, 0))
                new_image.paste(rgb_pil_image, (input_tensor.shape[3], 0))
                new_image.paste(cam_pil_image, (input_tensor.shape[3]*2, 0))
            new_image.save(file_path)
        return file_path

    def calculate_cam_for_result_table(self, result_table):
        """Calculate grad_cam for entire result table, result file paths are stored in 'cam_path'
        :param result_table: table where intermediate results are stored
        :return: modified result_table, results are stored in 'cam_path'"""
        result_table = result_table.sort_index()
        cam_path = []
        model_name = os.path.basename(self.m_config.get("trained_model_path"))
        for index, row in result_table.iterrows():
            cam_path.append(self.calculate_cam(row["file_path"], model_name))
        result_table["cam_path"] = cam_path
        return result_table

    def visualize(self, result_table, embeddings):
        """Visualize results with plotly_visulaizer, run on localhost in dev mode
        :param result_table: result table, can be calulated with `run_inference`
        :param embeddings: embeddings, can be calulated with 'get_embeddigns', can be None"""
        import plotly_visualizer
        from sklearn.manifold import TSNE
        from sklearn.metrics import classification_report
        pred_encoding = cm.get_predictions(np.array(result_table["true_class"]), np.array(result_table["pred_class"]))
        result_table["predictions"] = pred_encoding
        acc = accuracy_score(result_table["true_class"], result_table["pred_class"])
        f1score = f1_score(result_table["true_class"], result_table["pred_class"], average='weighted')
        scatter = plotly_visualizer.create_scatter_plot(
            result_table,
            "pred_class",
            "pred_prob",
            "predictions",
            "cam_path",
            "Acc: {0}, F1_score: {1}".format(acc, f1score),
        )
        labels = []
        with open(os.path.join(self.m_config.get("trained_model_path"), "labels.txt"), 'r') as my_file:
            for line in my_file:
                labels.append(line.replace('\n', ''))
        conf_matrix = plotly_visualizer.create_confustion_matrix(result_table, "Test", "true_class", "pred_class", labels)
        settings_text = plotly_visualizer.create_text_area(pprint.pformat(dict(self.m_config.items())))
        class_record = plotly_visualizer.create_text_area(classification_report(result_table['true_class'],result_table['pred_class'], target_names=labels), 1)
        figures = [scatter, conf_matrix,class_record, settings_text]
        if embeddings is not None:
            X_embedded = TSNE(n_components=2).fit_transform(embeddings)
            x, y = X_embedded[:, 0], X_embedded[:, 1]
            result_table["x_tsne"] = x
            result_table["y_tsne"] = y
            tsne = plotly_visualizer.create_scatter_plot(
                result_table, "x_tsne", "y_tsne", "true_class", "file_path", "T-SNE", 1
            )
            figures.append(tsne)
        plotly_visualizer.plot_html_figures(figures)
        plotly_visualizer.app.run_server(host="127.0.0.1", debug=False)
