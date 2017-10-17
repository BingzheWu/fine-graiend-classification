from options import opt
import torch
from torch.autograd import Variable
from utils import AverageMeter, accuracy
import time
classes = ["a", "ss", "gs", "cc", "fcc", "fc", "nos",]
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

def eval_class(net, opt, testLoader):
    class_correct = list(0. for i in range(opt.num_classes))
    class_total = list(0. for i in range(opt.num_classes))
    for data in testLoader:
        images, targets = data
        if opt.use_cuda:
            images = images.cuda()
            targets = targets.cuda()
        outputs = net(Variable(images))
        _, predict = torch.max(outputs.data, 1)
        c = (predict == targets).squeeze()
        for i in range(targets.size(0)):
            label = targets[i]
            class_correct[label] += c[i]
            class_total[label] += 1
    prec1 = 0
    for i in range(opt.num_classes):
        prec1 += 100*class_correct[i] / (class_total[i] * float(opt.num_classes))   
        print(('Accuracy of %5s : %2d %%')%(classes[i], 100*class_correct[i]/class_total[i]))
    return prec1