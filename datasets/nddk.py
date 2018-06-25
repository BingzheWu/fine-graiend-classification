import torch.utils.data as data
import torchvision
from dataset_utils import *
import os
import torch

tag2id = {"a":0,
        "ss":0,
        "gs":0,
        "cc":0,
        "fcc":0,
        "fc":0,
        "nos":1,
        }
def make_dataset(dir, mode = 's_nos'):
    """
    return the list of all image paths
    """
    images = []
    num_s = 0
    num_nos = 0
    num_gs = 0
    num_ss = 0
    num_cc = 0
    num_fc = 0
    num_fcc = 0
    patient_ids = {}
    for image_file in os.listdir(dir):
        if is_image_file(image_file):
            path = os.path.join(dir, image_file)
            tag, target = extract_class_label(image_file)
            if mode == 'ss_gs':
                if tag !='ss' and tag != 'gs':
                    continue
                if tag == 'ss':
                    num_ss+=1
                else:
                    num_gs+=1
            if mode == 's_nos':
                if tag == 'a' or 'c' in tag:
                    continue
                if tag == 'nos':
                #    if num_nos >= 4000:
                #        continue
                    num_nos +=1
                else:
                    num_s +=1
            if mode == 'c_s':
                if tag=='nos' or tag=='a':
                    continue
                if tag == 'cc':
                    num_cc += 1
                if tag =='fc':
                    num_fc+=1
                if tag == 'fcc':
                    num_fcc+=1
            if mode == 'c':
                if 'c' not in tag:
                    continue
            if mode == 'c_nos':
                if 'c' not in tag:
                    if tag =='nos':
                        num_nos+=1
                        if num_nos >=1000:
                            continue
                    else:
                        continue
            if mode == 'cs_nos':
                if tag == 'a':
                    continue
                if tag == 'nos':
                    if num_nos > 7000:
                        continue
                    num_nos+=1   
            if mode == 'all':
                if tag == 'a':
                    continue
                if tag == 'nos':
                    if num_nos > 3000:
                        continue
                    num_nos+=1
            tag, target = extract_class_label(image_file)
            item = (path, target)
            images.append(item)
    print("nos:%d"%(num_nos))
    print("s:%d"%(num_s))
    print("ss:%d"%(num_ss))
    print("gs:%d"%(num_gs))
    print("cc:%d"%(num_cc))
    print("fc:%d"%(num_fc))
    print("fcc:%d"%(num_fcc))
    return images
def extract_class_label(fname):
    """
    extract the label of the Glomerulus image
    """
    prefix = os.path.splitext(fname)[0]
    tag = prefix.split("_")[-1].lower()
    return tag,tag2id[tag]
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
        transform = trans.Compose([trans.Resize(self.opt.imageSize), trans.RandomHorizontalFlip(), trans.RandomVerticalFlip(), trans.ToTensor()])
        #transform = trans.Compose([trans.Resize(self.opt.imageSize), trans.ToTensor()])
        return transform
    def __len__(self):
        return self.img_num

    
class NCKD_fake(data.Dataset):
    """
    opt: options for data settings
    """
    def __init__(self, opt, transform = None, target_transform = None,
            loader = pil_loader, is_train = True):
        super(NCKD_fake, self).__init__()
        self.imgs =  make_dataset(opt.dataroot)
        self.img_num = len(self.imgs)
        self.opt = opt
        self.transform = self.transform()
        self.target_transform = target_transform
        self.loader = loader
        self.is_train = opt.is_train
    def __getitem__(self, index):
        path, target = self.imgs[index]
        if self.is_train:
            fake_path = path.replace('train_pas','train_fake_pasm')
        else:
            fake_path = path.replace('test_pas', 'test_fake_pasm')
        fake_path = fake_path.replace('.jpg', '_fake_B.png')
        img = self.loader(fake_path)
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
            loader = pil_loader, is_train = True, prefix = 'pasm'):
        super(NCKD_TWIN, self).__init__()
        self.imgs =  make_dataset(opt.dataroot)
        self.img_num = len(self.imgs)
        self.opt = opt
        self.transform = self.transform()
        self.target_transform = target_transform
        self.loader = loader
        self.is_train = opt.is_train
        self.prefix = prefix
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
        transform = trans.Compose([trans.Scale(self.opt.imageSize), trans.RandomHorizontalFlip(), trans.RandomVerticalFlip(), trans.ToTensor()])
        return transform
    def __len__(self):
        return self.img_num

class NCKD_quadruplets(data.Dataset):
    def __init__(self, opt, transform = None, target_transform = None,
            loader = pil_loader, is_train = True):
        super(NCKD_quadruplets, self).__init__()
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
            masson_fake_path = path.replace('train_pas', 'train_fake_masson')
            he_fake_path = path.replace('train_pas', 'train_fake_he')
            pasm_fake_path = path.replace('train_pas', 'train_fake_pasm')
        else:
            masson_fake_path = path.replace('test_pas', 'test_fake_masson')
            he_fake_path = path.replace('test_pas', 'test_fake_he')
            pasm_fake_path = path.replace('test_pas', 'test_fake_pasm')
            #masson_fake_path = path.replace('test_pas', 'test_fake_masson')
        masson_fake_path = masson_fake_path.replace('.jpg', '_fake_B.png')
        he_fake_path = he_fake_path.replace('.jpg', '_fake_B.png')
        pasm_fake_path = pasm_fake_path.replace('.jpg', '_fake_B.png')
        fake_path = [masson_fake_path, he_fake_path, pasm_fake_path]
        if self.transform is not None:
            img = self.transform(img)
        tmp =[img]
        for img_fake_path in fake_path: 
            img_fake = self.loader(img_fake_path)
            if self.transform is not None:
                img_fake = self.transform(img_fake)
            tmp.append(img_fake)
        if self.target_transform is not None:
            target = self.target_transform(target)
        img = torch.cat(tmp, dim = 0)
        return img, target
    def transform(self):
        trans = torchvision.transforms
        transform = trans.Compose([trans.Resize(self.opt.imageSize), trans.RandomHorizontalFlip(), trans.RandomVerticalFlip(), trans.ToTensor()])
        return transform
    def __len__(self):
        return self.img_num