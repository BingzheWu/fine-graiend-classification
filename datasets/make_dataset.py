import torch
from nddk import NCKD, NCKD_TWIN, NCKD_quadruplets, NCKD_fake
from IDC import IDC
import numpy as np
def make_weights_for_balanced_classes(images, nclasses):
    count = [0]*nclasses
    for item in images:
        count[item[1]] +=1
    weight_per_class = [0.]*nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0]*len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
        weight[idx] = 0.01
    return weight
def make_dataset(opt, is_train = True, mode = 'train', use_sampler = True, print_fail_img = False):
    """
    make data loader
    """
    if opt.dataset == 'NCKD':
        dataset = NCKD(opt)
    if opt.dataset == 'NCKD_TWIN':
        dataset = NCKD_TWIN(opt, prefix = opt.nckd_mode)
    if opt.dataset == 'NCKD_quadruplets':
        dataset = NCKD_quadruplets(opt)
    if opt.dataset == 'NCKD_fake':
        dataset = NCKD_fake(opt)
    if opt.dataset == 'IDC':
        dataset = IDC(opt, mode)
    class_sample_count = [648,2002,274,7000]
    weights = torch.Tensor(class_sample_count)
    weights = weights.double()
    weights = make_weights_for_balanced_classes(dataset.imgs, opt.num_classes)
    weights = torch.DoubleTensor(weights)
    weights = weights.double()
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    data_iter = torch.utils.data.DataLoader(dataset, batch_size = opt.batch_size,
            shuffle = False,
            sampler = sampler,
            num_workers = opt.num_workers)
    data_iter = torch.utils.data.DataLoader(dataset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers)
    if mode == 'test' or mode == 'val':
        data_iter = torch.utils.data.DataLoader(dataset, batch_size = opt.batch_size, 
                shuffle = False,
                num_workers = opt.num_workers)       
    '''
    if not is_train:
        data_iter = torch.utils.data.DataLoader(dataset, batch_size = opt.batch_size)
    '''
    if print_fail_img:
        return data_iter, dataset.imgs
    return data_iter