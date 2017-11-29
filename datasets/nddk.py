import torch.utils.data as data
import torchvision
from dataset_utils import *
import os
import torch

tag2id = {"a":0,
        "ss":0,
        "gs":0,
        "cc":1,
        "fcc":1,
        "fc":1,
        "nos":1,
        }
def make_dataset(dir, mode = 's_nos'):
    """
    return the list of all image paths
    """
    images = []
    for image_file in os.listdir(dir):
        if is_image_file(image_file):
            path = os.path.join(dir, image_file)
            if mode == 's':
                if 'ss' not in path and 'gs' not in path:
                    continue
            if mode == 's_nos':
                if 'a_' in path or 'c_' in path:
                    continue
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
            loader = pil_loader, is_train = True):
        super(NCKD, self).__init__()
        self.imgs =  make_dataset(opt.dataroot)
        self.img_num = len(self.imgs)
        self.opt = opt
        self.transform = self.transform()
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
    def transform(self):
        trans = torchvision.transforms
        transform = trans.Compose([trans.Scale(self.opt.imageSize), trans.ToTensor()])
        return transform
    def __len__(self):
        return self.img_num


class NCKD_TWIN(data.Dataset):
    def __init__(self, opt, transform = None, target_transform = None,
            loader = pil_loader, is_train = True):
        super(NCKD_TWIN, self).__init__()
        self.imgs =  make_dataset(opt.dataroot)
        self.img_num = len(self.imgs)
        self.opt = opt
        self.transform = self.transform()
        self.target_transform = target_transform
        self.loader = loader
        self.is_train = opt.is_train
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.is_train:
            img_fake_path = path.replace('trainA', 'train_fake_B')
        else:
            img_fake_path = path.replace('testA', 'test_fake_B')
        img_fake_path = img_fake_path.replace('.jpg', '_fake_B.png')
        #img_fake_path = img_fake_path.replace('')
        img_fake = self.loader(img_fake_path)
        if self.transform is not None:
            img = self.transform(img)
            img_fake = self.transform(img_fake)
        if self.target_transform is not None:
            target = self.target_transform(target)
        img_tmp = torch.FloatTensor(3, 225, 225)
        img_tmp = img_tmp.normal_(0,1)
        img = torch.cat([img,img_fake], dim = 0)
        return img, target
    def transform(self):
        trans = torchvision.transforms
        transform = trans.Compose([trans.Scale(self.opt.imageSize), trans.ToTensor()])
        return transform
    def __len__(self):
        return self.img_num