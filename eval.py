from options import opt
import torch
from torch.autograd import Variable
import sys
sys.path.append('./datasets')
from make_dataset import make_dataset
from models import model_creator
from options import opt
from utils import AverageMeter, accuracy
import os
import time
import numpy as np
classes = ["a", "ss", "gs", "cc", "fcc", "fc", "nos",]
classes = ["a", "s", "c", "nos"]
classes = ["a", "n"]
classes = ["s", "nos"]
classes = ["ss", "gs"]
#classes = ['s', 'c']
#classes = ['cc', 'fc', 'fcc']
#classes = ['cs', 'nos']
classes = ["ss", "gs", "c", "nos"]
classes = ['cs', 'nos']
classes_dict = {'IDC':['n', 'p'], 'NCKD':['cs', 'nos'], 'NCKD_quadruplets':['cs', 'nos'], 'NCKD_TWIN':['ss', 'gs']}
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
    all_results = []
    all_labels = []
    classes = classes_dict[opt.dataset]
    for i, data in enumerate(testLoader):
        images, targets = data
        if opt.use_cuda:
            images = images.cuda()
            targets = targets.cuda()
        outputs = net(Variable(images))
        _, predict = torch.max(outputs.data, 1)
        c = (predict == targets).squeeze()
        if i == 0:
            all_results = outputs.data.cpu().numpy()
            all_labels = targets.cpu().numpy()
        else:
            all_results = np.append(all_results, outputs.cpu().data.numpy(), axis = 0)
            all_labels = np.append(all_labels, targets.cpu().numpy(), axis = 0)
        for i in range(targets.size(0)):
            label = targets[i]
            class_correct[label] += c[i]
            class_total[label] += 1
    np.save(os.path.join(opt.experiments, 'predicts'), all_results)
    np.save(os.path.join(opt.experiments, 'labels'), all_labels)
    prec1 = 0
    total_correct_num = 0.0
    total_num = 0.0
    for i in range(opt.num_classes):
        prec1 += 100*class_correct[i] / (class_total[i] * float(opt.num_classes))
        total_correct_num += class_correct[i]
        total_num += class_total[i]
        print(('Accuracy of %5s : %3f %%')%(classes[i], 100*class_correct[i]/class_total[i]))
    balanced_accuracy = 50*class_correct[0]/class_total[0] + 50*class_correct[1]/class_total[1]
    prec1 = float(100*total_correct_num/total_num)
    print("Total Accuracy %3f %%"%(balanced_accuracy))
    return balanced_accuracy
def eval_fail_case(net, opt, testLoader, testDataset):
    class_correct = list(0. for i in range(opt.num_classes))
    class_total = list(0. for i in range(opt.num_classes))
    all_results = []
    all_labels = []
    classes = classes_dict[opt.dataset]
    for i, data in enumerate(testLoader):
        images, targets = data
        if opt.use_cuda:
            images = images.cuda()
            targets = targets.cuda()
        outputs = net(Variable(images))
        _, predict = torch.max(outputs.data, 1)
        c = (predict == targets).squeeze()
        if i == 0:
            all_results = outputs.data.cpu().numpy()
            all_labels = targets.cpu().numpy()
        else:
            all_results = np.append(all_results, outputs.cpu().data.numpy(), axis = 0)
            all_labels = np.append(all_labels, targets.cpu().numpy(), axis = 0)
        fail_case = []
        for j in range(targets.size(0)):
            label = targets[j]
            class_correct[label] += c[j]
            class_total[label] += 1
            if c[0]==0 and label ==1:
                print(testDataset[i][0])
                fail_case.append(i)
    np.save('fail_pas_all', np.array(fail_case))
    np.save(os.path.join(opt.experiments, 'predicts'), all_results)
    np.save(os.path.join(opt.experiments, 'labels'), all_labels)
    prec1 = 0
    total_correct_num = 0.0
    total_num = 0.0
    for i in range(opt.num_classes):
        prec1 += 100*class_correct[i] / (class_total[i] * float(opt.num_classes))
        total_correct_num += class_correct[i]
        total_num += class_total[i]
        print(('Accuracy of %5s : %3f %%')%(classes[i], 100*class_correct[i]/class_total[i]))
    balanced_accuracy = 50*class_correct[0]/class_total[0] + 50*class_correct[1]/class_total[1]
    prec1 = float(100*total_correct_num/total_num)
    print("Total Accuracy %3f %%"%(balanced_accuracy))
    return balanced_accuracy
def main(opt):
    opt.dataroot = opt.testroot
    #opt.is_train = False
    test_loader,dataset = make_dataset(opt, mode = 'test')
    net = model_creator(opt)
    model_dict = torch.load(os.path.join(opt.experiments, 'model_best.pth.tar'))
    net = net.cuda()
    net = net.eval()
    net.load_state_dict(model_dict['state_dice'])
    net = net.eval()
    eval_fail_case(net, opt, test_loader, dataset)
if __name__ == '__main__':
    main(opt)
