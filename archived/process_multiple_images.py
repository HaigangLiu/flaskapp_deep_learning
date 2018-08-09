import torch, os
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compute_multilabel_AUCs(ground_truth, pred_prob, N_CLASSES):
    '''
    Args:
        input of groud truth and pred prob are numpy arrays.
        Each column corresponds each image, and column is the class.
    Return:
        A list of auc for each group.
    '''
    AUROCs = []
    for i in range(N_CLASSES):
        try:
            AUROCs.append(roc_auc_score(ground_truth[:, i], pred_prob[:, i]))
        except ValueError:
            AUROCs.append(0)
    return AUROCs

class MultipleImageHandler:
    '''
    This is a trimmed down and thus portable version.
    Used as the backend of multiple image module in the app.
    Args:
        path_cached_model: the path to the pytorch model
        dataloader: a pytorch dataloader object
    '''
    def __init__(self, path_cached_model, dataloader):

        self.model = torch.load(path_cached_model, map_location='cpu')
        self.dataloader = dataloader

    def predict(self):
        self.model.eval()
        probs = []; truth = []
        for images, labels in self.dataloader:

            images = images.to(device)
            labels = labels.to(device)
            with torch.set_grad_enabled(mode = False):
                outputs = self.model(images)
                probs.append(outputs)
                truth.append(labels)

        likelihood = torch.cat(probs, 0).detach().cpu().numpy()
        ground_truth = torch.cat(truth, 0).detach().cpu().numpy()

        return likelihood, ground_truth

class MultipleImageDataset(Dataset):
    def __init__(self, data_dir, image_list_file, transform = None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        image_names = []
        labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name= items[0]
                label = items[1:]
                label = [int(i) for i in label]
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)

def pipeline_for_multiple_images(cached_model_dir, image_folder_dir, groundtruth_dir, number_of_class = 14 ):


    NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    dataset= MultipleImageDataset(data_dir = image_folder_dir,
                                  image_list_file = groundtruth_dir,
                                  transform = transforms.Compose([transforms.Resize([224,224]),transforms.ToTensor(),NORMALIZE])
                                  )

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers = 4, drop_last=True)
    likelihood, ground_truth = MultipleImageHandler(cached_model_dir, dataloader).predict()
    roc_per_class = compute_multilabel_AUCs(ground_truth, likelihood, number_of_class)
    roc_mean = np.array(roc_per_class).mean()

    return roc_per_class, roc_mean

if __name__ == '__main__':
    image_folder_dir = '/Users/haigangliu/ImageData/ChestXrayData'
    groundtruth_dir= '/Users/haigangliu/flask_deep_learning/multiple_images/test_list_shorter.txt'
    model_dr = '/Users/haigangliu/flask_deep_learning/static/models/multi_class_.pth.tar'
    s = pipeline_for_multiple_images(model_dr,image_folder_dir, groundtruth_dir)
    print(s)
