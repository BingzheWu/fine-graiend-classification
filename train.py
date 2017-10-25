import torch
import torch.nn.functional as F
from torch.autograd import Variable
from models import model_creator
from options import opt
import torch.optim as optim
import sys
import os
sys.path.append('./datasets')
from make_dataset import make_dataset
from eval import eval_class
from utils import save_checkpoint, accuracy, AverageMeter
import time
def train(opt):
    if opt.trainroot:
        opt.dataroot = opt.trainroot
        train_loader = make_dataset(opt)
    if opt.testroot:
        opt.dataroot = opt.test_loader
        opt.is_train = False
        test_loader = make_dataset(opt)
    net = model_creator(opt)
    if opt.use_cuda:
        print("load cuda model")
        #net = torch.nn.DataParallel(net).cuda()
        net = net.cuda()
    if opt.resume:
        if os.path.isfile(opt.resume):
            checkpoint = torch.load(opt.resume)
            net.load_state_dict(checkpoint['state_dict'])
    losses = AverageMeter()
    top1 = AverageMeter()
    batch_time = AverageMeter()
    optimizer = optim.SGD(net.parameters(), lr = opt.lr, momentum = 0.9)
    best_prec1 = 0
    for epoch in range(opt.epochs):
        net.train()
        end = time.time()
        for batch_idx, (image, target) in enumerate(train_loader):
            image = Variable(image)
            target = Variable(target)
            if opt.use_cuda:
                image = image.cuda()
                target = target.cuda()
            logits = net(image)
            loss = F.cross_entropy(logits, target)
            prec1 = accuracy(logits.data, target.data)[0]
            losses.update(loss.data[0], image.size(0))
            top1.update(prec1[0], image.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_time.update(time.time()-end)
            end = time.time()
            if batch_idx % 50 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, batch_idx, len(train_loader), batch_time=batch_time,
                      loss=losses, top1=top1))
        print("start validate")
        prec1 = eval_class(net,opt, test_loader)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({'epoch': epoch+1,
        'state_dice': net.state_dict(),
        'best_prec1': best_prec1}, is_best, dir_name = opt.experiments)
    print('Best Accuracy:')
    print(best_prec1)
        
if __name__ == '__main__':
    train(opt)
