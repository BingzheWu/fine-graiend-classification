import torch.utils.data as data

from utils import *
import os

tag2id = {"a":0,
        "ss":1,
        "gs":1,
        "cc":1,
        "fcc":1,
        "fc":1,
        "nos":2,
        "normal":3
        }
def make_dataset(dir):
    """
    return the list of all image paths
    """
    images = []
    for image_file in os.list(dir):
        if is_image_file(image_file):
            path = os.path.join(dir, image_file)
            target = extract_class_label(image_file)
            item = (path, target)
            images.append(item)
    return images
def extract_class_label(fname):
    """
    extract the label of the Glomerulus image
    """
    prefix = os.path.splitext(fname)[0]
    tag = prefix.split("_")[-1].lower()
    return tag2id[tag]
class NCKD(data.Dataset):
    """
    opt: options for data settings
    """
    def __init__(self, opt, transform = None, target_transform = None,
            loader = None, is_train = True):
        super(self, NCKD).__init__()
        self.imgs =  make_dataset(opt.dataroot)
        self.img_num = len(self.imgs)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.is_train = is_train
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    def __len__(self):
        return self.img_num
