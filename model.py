import torch
import torchvision
import torch.nn.functional as F
from abc import ABCMeta
from abc import abstractmethod
import numpy as np
import tensorflow as tf
import gc
import numpy as np
import pandas as pd
import torch
from torchvision import transforms, datasets
from pathlib import Path
from types import SimpleNamespace
from PIL import Image
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelWrapper(object):
    __metaclass__ = ABCMeta
    MODEL_DICT = {
        "resnet50":224,
        "resnet152": 224,
        "densenet121":224,
        "inceptionv3":  299
    }
    @abstractmethod
    def __init__(self):
        self.model = None
        self.model_name = None
        self.transforms = self.get_transforms()

    def get_transforms(args=None):


        """Get appropriate transforms based on model type with controlled probabilities.
        Currently only standard settings implemented."""
        if args is None:
            home = str(Path.home())
            args=  SimpleNamespace(
                data_dir=f"{home}/datasets/EYEPACS_prepr",
                label_file=f"{home}/datasets/EYEPACS_labels.txt",
                flip_prob=0.5,
                max_rotation_degree=180,
                use_rotation=True,
                use_dataset_stats=True
            )


        def compute_dataset_stats(dataset):
            """Compute mean and std for dataset normalization."""
            loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
            mean = 0.0
            std = 0.0
            n_samples = 0

            for images, _ in loader:
                batch_samples = images.size(0)
                images = images.view(batch_samples, images.size(1), -1)  # Flatten per channel
                mean += images.mean(dim=[0, 2]) * batch_samples
                std += images.std(dim=[0, 2]) * batch_samples
                n_samples += batch_samples

            mean /= n_samples
            std /= n_samples
            return mean.numpy(), std.numpy()

        def get_dataset_stats(data_dir):
            """Get dataset statistics for normalization."""
            dataset_dir = os.path.dirname(data_dir)
            path_data_stats = os.path.join(dataset_dir, "FGADR_data_stats.csv")
            
            if not os.path.exists(f"{path_data_stats}/FGADR_data_stats.csv"):
                print(f"ERROR: No dataset statistics in {path_data_stats}/FGADR_data_stats.csv")
            else:
                dataset_stats = pd.read_csv(f"{path_data_stats}/FGADR_data_stats.csv")
                mean = dataset_stats["mean"].values
                std = dataset_stats["std"].values
            return mean, std
        
        
        model_input_sizes = ModelWrapper.MODEL_DICT.values()
        
        img_size = model_input_sizes.get(args.model_name.lower(), 224)
        
        transform_list = [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=args.flip_prob),
            transforms.RandomVerticalFlip(p=args.flip_prob)
        ]
        
        if args.use_rotation:
            transform_list.append(transforms.RandomRotation(degrees=(0, args.max_rotation_degree), fill=0))  # Ensuring black padding
        if args.use_dataset_stats:
            mean, std = get_dataset_stats(args.data_dir, args.model_name)
        else:
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        return transforms.Compose(transform_list)

    @abstractmethod
    def get_cutted_model(self, bottleneck):
        pass

    def get_gradient(self, acts, y, bottleneck_name):
        acts = np.array(acts)
        inputs = torch.autograd.Variable(torch.tensor(acts).to(device), requires_grad=True)
        targets = (y[0] * torch.ones(inputs.size(0))).long().to(device)

        cutted_model = self.get_cutted_model(bottleneck_name).to(device)
        cutted_model.eval()
        outputs = cutted_model(inputs)

        # y=[i]
        grads = -torch.autograd.grad(outputs[:, y[0]], inputs)[0]
        
        grads = grads.detach().cpu().numpy()

        cutted_model = None
        gc.collect()

        return grads

    def reshape_activations(self, layer_acts):
        return np.asarray(layer_acts).squeeze()

    @abstractmethod
    def label_to_id(self, label):
        pass

    def run_examples(self, examples, bottleneck_name):
        def transform_np2tensor(rgb_list, transform):
            """
            Applies the given transformation to a list of NumPy RGB images and converts them to a tensor.
            
            Args:
                rgb_list (list of np.ndarray): List of RGB images in float format (0-1) with shape (H, W, C).
                transform (torchvision.transforms.Compose): Transformation pipeline from get_transforms.

            Returns:
                torch.Tensor: Tensor of transformed images with shape (N, C, H, W).
            """
            transformed_images = [transform(np_to_pil(img)) for img in rgb_list]
            return torch.stack(transformed_images)

        def np_to_pil(np_img):
            """
            Converts a NumPy image (H, W, C) in float format (0-1) to a PIL Image.
            
            Args:
                np_img (np.ndarray): Input image.

            Returns:
                PIL.Image: Converted image.
            """
            return transforms.functional.to_pil_image((np_img * 255).astype(np.uint8))

        global bn_activation
        bn_activation = None

        def save_activation_hook(mod, inp, out):
            global bn_activation
            bn_activation = out
        handle = self.model._modules[bottleneck_name].register_forward_hook(save_activation_hook)

        self.model.to(device)
        # inputs = torch.FloatTensor(examples).permute(0, 3, 1, 2).to(device) #TODO F6D:  go make this b a dataloader
        inputs = transform_np2tensor(examples, self.transforms).to(device)
        self.model.eval()
        self.model(inputs)
        acts = bn_activation.detach().cpu().numpy()
        handle.remove()

        return acts


class ImageModelWrapper(ModelWrapper):
    """Wrapper base class for image models."""

    def __init__(self, image_shape):
        super(ModelWrapper, self).__init__()
        # shape of the input image in this model
        self.image_shape = image_shape

    def get_image_shape(self):
        """returns the shape of an input image."""
        return self.image_shape


class PublicImageModelWrapper(ImageModelWrapper):
    """Simple wrapper of the public image models with session object.
    """

    def __init__(self, labels_path, image_shape):
        super(PublicImageModelWrapper, self).__init__(image_shape=image_shape)
        self.labels = tf.io.gfile.GFile(labels_path, "r").read().splitlines()
        # print(f"Loaded labels: {self.labels}")

    def label_to_id(self, label):
        if isinstance(label, int):
            label = str(label)
        return self.labels.index(label)


class InceptionV3_cutted(torch.nn.Module):
    def __init__(self, inception_v3, bottleneck):
        super(InceptionV3_cutted, self).__init__()
        names = list(inception_v3._modules.keys())
        layers = list(inception_v3.children())

        self.layers = torch.nn.ModuleList()
        self.layers_names = []

        bottleneck_met = False
        for name, layer in zip(names, layers):
            if name == bottleneck:
                bottleneck_met = True
                continue  # because we already have the output of the bottleneck layer
            if not bottleneck_met:
                continue
            if name == 'AuxLogits':
                continue

            self.layers.append(layer)
            self.layers_names.append(name)

    def forward(self, x):
        y = x
        for i in range(len(self.layers)):
            # pre-forward process
            if self.layers_names[i] == 'Conv2d_3b_1x1':
                y = F.max_pool2d(y, kernel_size=3, stride=2)
            elif self.layers_names[i] == 'Mixed_5b':
                y = F.max_pool2d(y, kernel_size=3, stride=2)
            elif self.layers_names[i] == 'fc':
                y = F.adaptive_avg_pool2d(y, (1, 1))
                y = F.dropout(y, training=self.training)
                y = y.view(y.size(0), -1)

            y = self.layers[i](y)
        return y


class InceptionV3Wrapper(PublicImageModelWrapper):

    def __init__(self, labels_path, model_weights=None):
        image_shape = [299, 299, 3]
        super(InceptionV3Wrapper, self).__init__(image_shape=image_shape,
                                                 labels_path=labels_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torchvision.models.inception_v3(pretrained=(model_weights is None)).to(device)#, transform_input
        
        if model_weights:
            state_dict = torch.load(model_weights, map_location=torch.device(device))
            self.model.load_state_dict(state_dict)
        self.model_name = 'inceptionv3'

    def forward(self, x):
        return self.model.forward(x)

    def get_cutted_model(self, bottleneck):
        return InceptionV3_cutted(self.model, bottleneck)



class ResNet50_cutted(torch.nn.Module):
    def __init__(self, resnet50, bottleneck):
        super(ResNet50_cutted, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers_names = []
        
        # Extract all layers
        names = list(resnet50._modules.keys())
        layers = list(resnet50.children())

        bottleneck_met = False
        for name, layer in zip(names, layers):
            if name == bottleneck:
                bottleneck_met = True
                continue  # Stop at the bottleneck layer
            if not bottleneck_met:
                continue

            self.layers.append(layer)
            self.layers_names.append(name)

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)

            if isinstance(self.layers[i], torch.nn.AdaptiveAvgPool2d):  
                x = torch.flatten(x, start_dim=1)
        return x


class ResNet50Wrapper(PublicImageModelWrapper):
    def __init__(self, labels_path, model_weights = None):
        image_shape = [224, 224, 3]  # ResNet50 standard input size
        super(ResNet50Wrapper, self).__init__(image_shape=image_shape, labels_path=labels_path)
        #Load the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torchvision.models.resnet50(pretrained=(model_weights is None)).to(device)
        if model_weights:
            state_dict = torch.load(model_weights, map_location=torch.device(device))
            self.model.load_state_dict(state_dict)

        self.model_name = 'resnet50'

    def forward(self, x):
        return self.model(x)

    def get_cutted_model(self, bottleneck):
        return ResNet50_cutted(self.model, bottleneck)

class ResNet152_cutted(torch.nn.Module):
    def __init__(self, resnet152, bottleneck):
        super(ResNet152_cutted, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers_names = []
        
        names = list(resnet152._modules.keys())
        layers = list(resnet152.children())

        bottleneck_met = False
        for name, layer in zip(names, layers):
            if name == bottleneck:
                bottleneck_met = True
                continue  # Stop at the bottleneck layer
            if not bottleneck_met:
                continue
            self.layers.append(layer)
            self.layers_names.append(name)

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if isinstance(self.layers[i], torch.nn.AdaptiveAvgPool2d):  
                x = torch.flatten(x, start_dim=1)
        return x


class ResNet152Wrapper(PublicImageModelWrapper):
    def __init__(self, labels_path, model_weights=None):
        image_shape = [224, 224, 3]  # ResNet152 standard input size
        super(ResNet152Wrapper, self).__init__(image_shape=image_shape, labels_path=labels_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torchvision.models.resnet152(pretrained=(model_weights is None)).to(device)
        if model_weights:
            state_dict = torch.load(model_weights, map_location=torch.device(device))
            self.model.load_state_dict(state_dict)

        self.model_name = 'resnet152'

    def forward(self, x):
        return self.model(x)

    def get_cutted_model(self, bottleneck):
        return ResNet152_cutted(self.model, bottleneck)


class DenseNet121_cutted(torch.nn.Module):
    def __init__(self, densenet121, bottleneck):
        super(DenseNet121_cutted, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers_names = []
        
        names = list(densenet121._modules.keys())
        layers = list(densenet121.children())

        bottleneck_met = False
        for name, layer in zip(names, layers):
            if name == bottleneck:
                bottleneck_met = True
                continue  # Stop at the bottleneck layer
            if not bottleneck_met:
                continue

            self.layers.append(layer)
            self.layers_names.append(name)

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if isinstance(self.layers[i], torch.nn.AdaptiveAvgPool2d):  
                x = torch.flatten(x, start_dim=1)
        return x


class DenseNet121Wrapper(PublicImageModelWrapper):
    def __init__(self, labels_path, model_weights=None):
        image_shape = [224, 224, 3]  # DenseNet121 standard input size
        super(DenseNet121Wrapper, self).__init__(image_shape=image_shape, labels_path=labels_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torchvision.models.densenet121(pretrained=(model_weights is None)).to(device)
        if model_weights:
            state_dict = torch.load(model_weights, map_location=torch.device(device))
            self.model.load_state_dict(state_dict)

        self.model_name = 'densenet121'

    def forward(self, x):
        return self.model(x)

    def get_cutted_model(self, bottleneck):
        return DenseNet121_cutted(self.model, bottleneck)
