import os
import torch
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
USER_PATH = os.path.expanduser('~')

def _create_dataset(dataroot, dstype='train', shuffle=True):
    switch_name = { 'train': 'train-color', 'test': 'test-color', 'val': 'validate-color' }
    base_in_name = switch_name[dstype]
    base_in_by_class = [
            os.path.join(dataroot, '{}/yaw01').format(base_in_name), # front
            os.path.join(dataroot, '{}/yaw05').format(base_in_name), # driverside
            os.path.join(dataroot, '{}/yaw09').format(base_in_name), # back
            os.path.join(dataroot, '{}/yaw13').format(base_in_name) # passengerside
            ]
    labeled_img_paths = []
    for i in range(len(base_in_by_class)):
        base_in = os.path.join(USER_PATH, base_in_by_class[i])
        for f in os.listdir(base_in):
            if f.startswith('.'):
                continue
            full_path = os.path.join(base_in, f)
            labeled_img_paths.append((i, full_path))
    if shuffle:
        np.random.shuffle(labeled_img_paths)
    return labeled_img_paths

class CarvanaDataset(data.Dataset):
    def __init__(self):
        super(CarvanaDataset, self).__init__()

    def initialize(self, opt):
        self._dat = _create_dataset(opt.dataroot, dstype=opt.phase)
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        target, path = self._dat[index]
        X = Image.open(path).convert('RGB')
        X = self.transform(X)
        return X, torch.LongTensor([target])

    def __len__(self):
        return len(self._dat)

    def name(self):
        return 'CarvanaDataset'
