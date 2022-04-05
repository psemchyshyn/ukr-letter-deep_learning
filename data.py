import glob
from PIL import Image
import os
from torch.utils.data.dataset import Dataset


class UKR_LETTERS_DS(Dataset):
  def __init__(self, path, transforms=None):
    self.items = glob.glob(path + "/*/*")
    self.transforms = transforms
    self.label_to_ind(path)

  def label_to_ind(self, path):
    labels = []
    for i in glob.glob(path + "/*"):
      label = i.split(os.sep)[1][0]
      labels.append(label)
    self.label_ind_dct = {label: i for i, label in zip(range(len(labels)), labels)}
    self.ind_label_dct = {i: label for label, i in self.label_ind_dct.items()}

  def __getitem__(self, idx):
    image_path = self.items[idx]
    image = Image.open(image_path).convert("L")
    class_label = image_path.split(os.sep)[1][0]
    return self.transforms(image), self.label_ind_dct[class_label]

  def __len__(self):
    return len(self.items)


class ANOMALY_DS(Dataset):
    def __init__(self, data_path, anomaly_class, transforms=None):
        self.items = glob.glob(f"{data_path}/{anomaly_class}_1/*")
        self.transforms = transforms


    def __getitem__(self, idx):
        image_path = self.items[idx]
        image = Image.open(image_path).convert("L")
        class_label = image_path.split(os.sep)[1][0]
        return self.transforms(image)

    def __len__(self):
        return len(self.items)

