import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class CIFARDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        img = torch.tensor(img, dtype=torch.float) / 255.0  # shape: (3,32,32)

        if self.transform:
            from torchvision.transforms.functional import to_pil_image
            img = to_pil_image(img)
            img = self.transform(img)

        return img, label


def unpickle(file):
    with open(file, 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')
    return dict_


def load_cifar10_train(data_path):
    all_data = []
    all_labels = []
    for i in range(1, 6):
        batch_file = os.path.join(data_path, f"data_batch_{i}")
        batch_dict = unpickle(batch_file)
        data = batch_dict[b'data']
        labels = batch_dict[b'labels']
        all_data.append(data)
        all_labels.extend(labels)
    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.array(all_labels)
    return all_data, all_labels


def load_cifar10_val(data_path):
    val_dict = unpickle(os.path.join(data_path, "test_batch"))
    val_data = val_dict[b'data']
    val_labels = val_dict[b'labels']
    return val_data, np.array(val_labels)


def convert_data_shape(data_array, channels=3, img_size=32):
    N = data_array.shape[0]
    data_array = data_array.reshape((N, channels, img_size, img_size))
    return data_array
