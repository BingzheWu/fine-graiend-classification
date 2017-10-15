import torch
from nddk import NCKD, NCKD_TWIN

def make_dataset(opt):
    """
    make data loader
    """
    if opt.dataset == 'NCKD':
        dataset = NCKD(opt)
    if opt.dataset == 'NCKD_TWIN':
        dataset = NCKD_TWIN(opt)
    data_iter = torch.utils.data.DataLoader(dataset, batch_size = opt.batch_size,
            shuffle = True, 
            num_workers = opt.num_workers)
    return data_iter

