from options import opt
import torch
from torch.autograd import Variable
from utils import AverageMeter, accuracy
import time
def eval(net, opt, testLoader, topk = (1,)):
    """
    validate given dataset
    """
    net.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    for idx, (image, target) in enumerate(testLoader):
        if opt.use_cuda:
            image = image.cuda()
            target = target.cuda()
        image = Variable(image)
        logits = net(image)
        prec1 = accuracy(logits.data, target, topk)[0]
        top1.update(prec1[0], image.size(0))
        batch_time.update(time.time()-end)
        end = time.time()
    print(top1.avg)
    return top1.avg


        
