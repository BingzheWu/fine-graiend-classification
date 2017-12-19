import torch
from nddk import NCKD, NCKD_TWIN, NCKD_quadruplets, NCKD_fake

def make_dataset(opt):
    """
    make data loader
    """
    if opt.dataset == 'NCKD':
        dataset = NCKD(opt)
    if opt.dataset == 'NCKD_TWIN':
        dataset = NCKD_TWIN(opt)
    if opt.dataset == 'NCKD_quadruplets':
        dataset = NCKD_quadruplets(opt)
    if opt.dataset == 'NCKD_fake':
        dataset = NCKD_fake(opt)
    data_iter = torch.utils.data.DataLoader(dataset, batch_size = opt.batch_size,
            shuffle = True, 
            num_workers = opt.num_workers)
    return data_iter

