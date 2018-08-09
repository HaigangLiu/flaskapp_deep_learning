import cv2
from torch.nn import functional as F
from matplotlib.pyplot import imshow
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import time
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import Dataset, DataLoader
from PIL import Image

NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

class SingleImageHandler():
    """
    Generate a prediction for a single image, and provide a CAM visual at the same tim.e

    CAM: Class Activation Map for multi-classes
    Generate a heatmap on top of the existing image
    To help visualize the convolutional nueral network.
    For more details please see Zhou et. al (2016)

    Args:
        model: pytorch model object, a trained model
        name_of_the_last_conv_layer (str): The name of the layer to visualize
    """
    def __init__(self, model, name_of_the_last_conv_layer, specialized_dataloader):

        self.model = model
        self.name_of_the_last_conv_layer = name_of_the_last_conv_layer
        self.specialized_dataloader = specialized_dataloader
        self._register()

    def _register(self):
        def forward_recorder(module, input, output):
            self.feature_maps = output.data.cpu()

        def backward_recorder(module, grad_in, grad_out):
            self.gradient = grad_out[0].data.cpu()

        for i, j in self.model.named_modules():
            if i ==  self.name_of_the_last_conv_layer:
                try:
                    self.length_of_filter = j.out_channels
                    self.size_of_kernel = j.kernel_size
                except AttributeError:
                    print("Target layer is not conv2d layer")
                j.register_forward_hook(forward_recorder)
                j.register_backward_hook(backward_recorder)
                break
        else:
            print("Cannot find the given layer, maybe try another name.")
            return None

    def process_one_image(self, image_tensor):
        self.image_tensor = image_tensor[0]
        self.image_numpy = self.image_tensor[0].numpy().transpose(1,2,0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        self.image_numpy = std * self.image_numpy + mean
        self._forward_pass()
        _, preds = torch.max(self.output, 1)
        self._backward_pass(which_class = preds)

    def _forward_pass(self):
        self.output = self.model(self.image_tensor)

        batch_size, self.channels, self.size_of_feature_maps, _ = self.feature_maps.size()
        try:
            self.feature_maps = self.feature_maps.squeeze(0)
        except:
            raise ValueError('Make sure setting the batch size to 1.')

    def _backward_pass(self, which_class):
        self.target_class = which_class
        batch_size, one_hot_length = self.output.size()

        one_hot = torch.FloatTensor(1, one_hot_length).zero_()
        one_hot[0][self.target_class] = 1.0
        self.output.backward(one_hot, retain_graph = True)

    def find_gradient_CAM(self):
        self.gradient = self.gradient/(torch.sqrt(torch.mean(torch.pow(self.gradient, 2))) + 1e-5)
        batch_size, channels, size_of_gradient, _ = self.gradient.size()
        gradient_average = nn.AvgPool2d([size_of_gradient, size_of_gradient])(self.gradient) #get mean gradient for each layer
        assert self.length_of_filter == channels, "Number of filter does not match gradient dimension"
        gradient_average.resize_(channels) # lots of squeeze

        self.gradient_CAM = torch.FloatTensor(self.size_of_feature_maps, self.size_of_feature_maps).zero_() # 7 by 7 probably
        for feature_map, weight in zip(self.feature_maps, gradient_average):
                self.gradient_CAM = self.gradient_CAM + feature_map * weight.data

        self.gradient_CAM = SingleImageHandler._post_processing_for_cam(self.gradient_CAM)

    @staticmethod
    def _post_processing_for_cam(gradient_CAM):

        gradient_CAM = F.leaky_relu(gradient_CAM)
        gradient_CAM = gradient_CAM - gradient_CAM.min();
        gradient_CAM = gradient_CAM/gradient_CAM.max()
        gradient_CAM = cv2.resize(gradient_CAM.numpy(), (224, 224))
        gradient_CAM = cv2.applyColorMap(np.uint8(gradient_CAM * 255.0), cv2.COLORMAP_JET)
        return gradient_CAM

    def save(self, out_dir):
        self.image_numpy = cv2.resize(self.image_numpy, (224, 224))
        combined_image = self.gradient_CAM.astype(np.float) + self.image_numpy*255
        gradient_CAM = (combined_image/combined_image.max())*255

        matplotlib.rcParams['axes.linewidth'] = 0.1
        fig = plt.figure(dpi = 200)
        ax1 = fig.add_subplot(1,2,1)
        ax1.imshow(self.image_numpy)
        ax2 = fig.add_subplot(1,2,2)
        ax2.imshow((gradient_CAM/255))
        fig.savefig(out_dir, bbox_inches="tight", pad_inches=0.2) #remove padding
        plt.close()

    def process(self):
        image = list(self.specialized_dataloader)
        self.process_one_image(image)
        self.find_gradient_CAM()
        return self

class OnePicDataset(Dataset):
    def __init__(self, root_dir, transform = None):
        self.root_dir = root_dir
        self.image_files = [file for file in os.listdir(self.root_dir) if
        file.endswith(('.png','jpeg','jpg'))]
        self.transform = transform

    def __len__(self):
        return 1
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx] )
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def generate_prediction(path_to_image_folder, path_to_model, new_image_name_abs):
    transform_val = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    dataset = OnePicDataset(path_to_image_folder,transform_val)
    one_image_loader = DataLoader(dataset = dataset,
                                  batch_size = 1,
                                  num_workers = 1)
    model = torch.load(path_to_model, map_location='cpu')
    cam_generator = SingleImageHandler(model,'features.denseblock4.denselayer16.conv2', one_image_loader)
    cam_generator.process().save(new_image_name_abs)
    return cam_generator.output.detach().numpy()[0]
