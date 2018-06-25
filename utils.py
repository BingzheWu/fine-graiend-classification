import torch
import os 
import shutil
import collections
import numpy as np
def accuracy(out, target, topk = (1,)):
    maxk = max(topk)
    batch_size = out.size(0)
    _, pred = out.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1,-1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res
def class_accuracy(out, target):
    pass
class AverageMeter(object):
    """
    computes and stores the average meter
    """
    def __init__(self, recent = None):
        self._recent = recent
        if recent is not None:
            self._q = collections.deque()
        self.reset()
    def reset(self):
        self.value = 0
        self.avg = 0
        self.count = 0
        self.sum = 0
        if self._recent is not None:
            self.sum_recent = 0
            self.count_recent = 0
            self._q.clear()
    def update(self, value, n = 1):
        self.value = value
        self.sum += value * n
        self.count += n
        if self._recent is not None:
            self.sum_recent += value * n
            self.count_recent += n
            self._q.append((n,value))
            while len(self._q) > self._recent:
                (n,value) = self._q.popleft()
                self.sum_recent -= value*n
                self.count_recent -= n
    @property
    def average(self):
        if self.count > 0:
            return self.sum / self.count
        else:
            return 0
    @property
    def average_recent(self):
        if self.count_recent > 0:
            return self.sum_recent / self.count_recent
        else:
            return 0
    
def topk_accuracy(outputs, labels, recalls=(1,)):
    _, num_classes = outputs.size()
    maxk = min(max(recalls), num_classes)
    _, pred = outputs.topk(maxk, dim = 1, largest=True, sorted = True)
    correct = (pred==labels[:,None].expand_as(pred)).float()
    topk_accuracy = []
    for recall in recalls:
        topk_accuracy.append(100*correct[:,:recall].sum(1).mean())
    return topk_accuracy
def classwise_accuracy(outputs, labels):
    batch_size, num_classes = outputs.size()
    score, pred_labels = torch.max(outputs, 1)
    pred_ = (pred_labels == labels).squeeze()
    classwise_num = np.zeros(num_classes)
    classwise_correct_num = np.zeros(num_classes)
    for i in range(batch_size):
        classwise_num[labels[i]] += 1
        if pred_[i]:
            classwise_correct_num[labels[i]]+=1
    return classwise_correct_num, classwise_num
        
def save_checkpoint(state, is_best, dir_name, filename = 'checkpoint.pth.tar'):
    filename = os.path.join(dir_name, filename)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(dir_name, 'model_best.pth.tar'))

