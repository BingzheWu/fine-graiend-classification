import os
import torch
from dataset_utils import *
import torch.utils.data as data
import torchvision

def make_dataset(dir, mode = 'train'):
    paient_id_file = 'cases_'+mode+'.txt'
    paient_id_file = os.path.join(dir, paient_id_file)
    num_n = 0
    num_p = 0
    images_labels = []
    with open(paient_id_file, 'r') as f:
        num_dict = {0:0, 1:0}
        for p_id in f.readlines():
            for tag in [0,1]:
                images_dir = os.path.join(dir, 'brust', p_id.strip(), str(tag).strip())
                if not os.path.exists(images_dir):
                    continue
                img_paths = os.listdir(images_dir)
                for img_path in img_paths:
                    path = os.path.join(images_dir, img_path)
                    item = (path, tag)
                    images_labels.append(item)
                    num_dict[tag] += 1
        print("num of n: %d"%(num_dict[0]))
        print("num of p: %d"%(num_dict[1]))
    return images_labels

class IDC(data.Dataset):
    def __init__(self, opt, mode = 'train',  loader = pil_loader):
        super(IDC, self).__init__()
        self.imgs = make_dataset(opt.dataroot, mode)
        self.img_num = len(self.imgs)
        self.mode = mode
        self.opt = opt
        self.transform_ = self.transform()
        self.loader = loader
        self.is_train = opt.is_train
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.mode == 'train':
            img = self.transform_(img)
        else:
            trans = torchvision.transforms
            transform_val = trans.Compose([trans.Resize((self.opt.imageSize, self.opt.imageSize)), trans.ToTensor(), trans.Normalize((0,0,0),(224,224,224))])
            transform_val = trans.Compose([trans.Resize((self.opt.imageSize, self.opt.imageSize)), trans.ToTensor()])
            img = transform_val(img)
        return img, target
    def transform(self):
        trans = torchvision.transforms
        #transform = trans.Compose([trans.Resize(self.opt.imageSize), trans.ToTensor()])
        transform = trans.Compose([trans.Resize((self.opt.imageSize, self.opt.imageSize)), trans.RandomHorizontalFlip(), trans.RandomVerticalFlip(), trans.ToTensor(), trans.Normalize((0,0,0),(224,224,224))])
        transform_val = trans.Compose([trans.Resize((self.opt.imageSize, self.opt.imageSize)), trans.ToTensor()])
        transform = trans.Compose([trans.Resize((self.opt.imageSize, self.opt.imageSize)), trans.RandomHorizontalFlip(), trans.RandomVerticalFlip(), trans.ToTensor()])
        return transform
    def __len__(self):
        return self.img_num
def test_make_dataset():
    base_dir = '/home/bingzhe/datasets/dp/'
    make_dataset(base_dir)

if __name__ == '__main__':
    test_make_dataset()
                