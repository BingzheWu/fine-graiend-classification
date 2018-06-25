import torch
import torch.nn.functional as F
from torch.autograd import Variable
import logging
from models import model_creator
from options import opt
import torch.optim as optim
import sys
import os
sys.path.append('./datasets')
from make_dataset import make_dataset
from eval import eval_class, test_for_one_epoch
import utils
from utils import save_checkpoint, accuracy, AverageMeter
import time
from noisy_sgd import dp_sgd
from loss import FocalLoss
def train(opt):
    if opt.trainroot:
        opt.dataroot = opt.trainroot
        train_loader = make_dataset(opt)
    if opt.testroot: 
        opt.dataroot = opt.testroot
        opt.is_train = False
        test_loader = make_dataset(opt, False, 'val')
    net = model_creator(opt)
    if opt.use_cuda:
        print("load cuda model")
        #net = torch.nn.DataParallel(net).cuda()
        #torch.cuda.set_device(2)
        net = net.cuda()
    if opt.resume:
        if os.path.isfile(opt.resume):
            checkpoint = torch.load(opt.resume)
            net.load_state_dict(checkpoint['state_dict'])
    losses = AverageMeter()
    top1 = AverageMeter()
    batch_time = AverageMeter()
    optimizer = optim.SGD(net.parameters(), lr = opt.lr, momentum = 0.9, weight_decay = 0.01)
    best_prec1 = 0
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [5,10,25,30,35,40], gamma = 0.05)
    for epoch in range(opt.epochs):
        net.train()
        end = time.time()
        lr_scheduler.step()
        print(len(train_loader))
        for batch_idx, data in enumerate(train_loader):
            image, target = data
            image = Variable(image)
            target = Variable(target)
            if opt.use_cuda:
                image = image.cuda()
                target = target.cuda()
            logits = net(image)
            focal_loss = F.cross_entropy(logits, target)
            focal_loss = FocalLoss(opt.num_classes)(logits, target)
            prec1 = accuracy(logits.data, target.data)[0]
            losses.update(focal_loss.data[0], image.size(0))
            top1.update(prec1[0], image.size(0))
            optimizer.zero_grad()
            focal_loss.backward()
            optimizer.step()
            batch_time.update(time.time()-end)
            end = time.time()
            if batch_idx % 50 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, batch_idx, len(train_loader), batch_time=batch_time,
                      losses=losses, top1=top1))
        print("start validate")
        prec1 = eval_class(net,opt, test_loader)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({'epoch': epoch+1,
        'state_dice': net.state_dict(),
        'best_prec1': best_prec1}, is_best, dir_name = opt.experiments)
    print('Best Accuracy:')
    print(best_prec1)
def train_for_one_epoch(net, loss, train_loader, optimizer, epoch_number):
    net.train()
    loss.train()
    data_time_meter = utils.AverageMeter()
    batch_time_meter = utils.AverageMeter()
    loss_meter = utils.AverageMeter(recent=100)
    top1_meter = utils.AverageMeter(recent=100)
    timestamp = time.time()
    for i, (images, labels) in enumerate(train_loader):
        batch_size = images.size(0)
        images = images.cuda(async = True)
        labels = labels.cuda(async = True)
        data_time_meter.update(time.time()-timestamp)

        outputs = net(images)
        loss_output = loss(outputs, labels)
        if isinstance(loss_output, tuple):
            loss_value, outputs = loss_output
        else:
            loss_value = loss_output
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 4.0)
        optimizer.step()
        optimizer.zero_grad()
        loss_meter.update(loss_value.item(), batch_size)
        top1 = utils.topk_accuracy(outputs, labels)[0]
        top1_meter.update(top1, batch_size)
        batch_time_meter.update(time.time()-timestamp)
        timestamp = time.time()
    logging.warning(
        'Epoch: [{epoch}] -- TRAINING SUMMARY\t'
        'Time {batch_time.sum:.2f}   '
        'Data {data_time.sum:.2f}   '
        'Loss {loss.average:.3f}     '
        'Top-1 {top1.average:.2f}    '.format(
            epoch=epoch_number, batch_time=batch_time_meter, data_time=data_time_meter,
            loss=loss_meter, top1=top1_meter))
def create_optimizer(net, momentum = 0.9, weight_decay = 0):
    model_traiable_parameters = filter(lambda x: x.requires_grad, net.parameters())
    optimizer = torch.optimizer.SGD(model_traiable_parameters, lr = 0, 
            momentum = momentum, weight_decay = weight_decay)
    return optimizer
def _set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def _get_learning_rate(optimizer):
    return max(param_group['lr'] for param_group in optimizer.param_groups)

def main():
    if opt.trainroot:
        opt.dataroot = opt.trainroot
        train_loader = make_dataset(opt)
    if opt.testroot: 
        opt.dataroot = opt.testroot
        opt.is_train = False
        test_loader = make_dataset(opt, False, 'val')
    net = model_creator(opt)
    if opt.use_cuda:
        print("load cuda model")
        net = net.cuda()
    if opt.resume:
        if os.path.isfile(opt.resume):
            checkpoint = torch.load(opt.resume)
            net.load_state_dict(checkpoint['state_dict'])
    losses = AverageMeter()
    top1 = AverageMeter()
    batch_time = AverageMeter()
    optimizer = optim.SGD(net.parameters(), lr = opt.lr, momentum = 0.9, weight_decay = 0.01)
    #optimizer = dp_sgd(net.parameters(), lr = opt.lr, momentum = 0.9, weight_decay = 0.01)
    #lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [5,10,25,30,35,40], gamma = 0.05)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(1, opt.epochs+1):
        #lr_scheduler.step()
        print(_get_learning_rate(optimizer))
        train_for_one_epoch(net, loss, train_loader, optimizer, epoch)
        if epoch % 2 == 1:
            test_for_one_epoch(net, loss, test_loader, epoch)

if __name__ == '__main__':
    main()
