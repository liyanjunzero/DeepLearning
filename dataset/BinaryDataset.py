import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

class BinaryDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        trans = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize(224)])
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = trans(image)
        label = torch.tensor(self.labels[idx])
        return image, label