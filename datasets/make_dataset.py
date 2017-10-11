import torch
from nddk import NCKD

def make_dataset(opt):
    """
    make data loader
    """
    if opt.dataset == 'NCKD':
        dataset = NCKD(opt)
    
    data_iter = torch.utils.data.Dataloader(dataset, batch_size = opt.batch_size,
            shuffle = True, 
            num_workers = opt.num_workers)
    return data_iter

