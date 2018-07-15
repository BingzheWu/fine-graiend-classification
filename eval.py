from options import opt
import torch
from torch.autograd import Variable
import sys
sys.path.append('./datasets')
from make_dataset import make_dataloader
from models import model_creator
from options import opt
from utils import AverageMeter, accuracy
import logging
import utils
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
classes_dict = {'IDC':['n', 'p'], 'NCKD':['cs', 'nos'], 'NCKD_quadruplets':['cs', 'nos'], 'NCKD_TWIN':['s', 'nos']}
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
    print(testDataset)
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
            if c[0]==0 and label ==0:	
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
def eval_by_id(net, opt, testLoader, testDataset):
    class_correct = list(0. for i in range(opt.num_classes))
    class_total = list(0. for i in range(opt.num_classes))
    all_results = []
    all_labels = []
    print(len(testDataset))
    patient_ids = {}
    patient_error = {}
    patient_total = {}
    classes = classes_dict[opt.dataset]
    for i, data in enumerate(testLoader):
        #print(testDataset[i][0].split('pas/')[1].split('_')[0])
        patient_id = testDataset[i][0].split('pas/')[1].split('_')[0]
        images, targets = data	
        if patient_id not in patient_total.keys():
            patient_error[patient_id] = 0
            patient_total[patient_id] = 0
            patient_ids[patient_id] = 0
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
            patient_ids[patient_id] += 1
            if c[0]==1 :
                patient_error[patient_id] += 1
    np.save('fail_pas_all', np.array(fail_case))	
    np.save(os.path.join(opt.experiments, 'predicts'), all_results)	
    np.save(os.path.join(opt.experiments, 'labels'), all_labels)	
    prec1 = 0	
    total_correct_num = 0.0	
    total_num = 0.0	
    acc_per_id = [float(patient_error[p_id])/ patient_ids[p_id] for p_id in patient_error.keys()]
    print("###############")
    print(acc_per_id)
    acc_per_id = np.array(acc_per_id)
    print(acc_per_id.mean())
    print(acc_per_id.std())
    print("################")
    balanced_accuracy = 50*class_correct[0]/class_total[0] + 50*class_correct[1]/class_total[1]	
    prec1 = float(100*total_correct_num/total_num)	
    print("Total Accuracy %3f %%"%(balanced_accuracy))
    return balanced_accuracy
def test_for_one_epoch(net, loss, test_loader, epoch_number, log_writer = None):
    net.eval()
    loss.eval()

    data_time_meter = utils.AverageMeter()
    batch_time_meter = utils.AverageMeter()
    loss_meter = utils.AverageMeter(recent = 100)
    top1_meter = utils.AverageMeter(recent=100)
    auc_meter = utils.AverageMeter(recent=100)
    timestamp = time.time()
    classwise_correct_num = np.zeros(2)
    classwise_num = np.zeros(2)
    for i, (images, labels) in enumerate(test_loader):
        batch_size = images.size(0)
        images = images.cuda(async = True)
        labels = labels.cuda(async = True)
        data_time_meter.update(time.time()-timestamp)
        with torch.no_grad():
            outputs = net(images)
            loss_output = loss(outputs, labels)
        if isinstance(loss_output, tuple):
            loss_value, outputs = loss_output
        else:
            loss_value = loss_output
        loss_meter.update(loss_value.item(), batch_size)
        top1 = utils.topk_accuracy(outputs, labels, recalls = (1, 5))[0]
        top1_meter.update(top1, batch_size)
        update1, update2 = utils.classwise_accuracy(outputs, labels)
        classwise_correct_num += update1
        classwise_num += update2
        batch_time_meter.update(time.time()-timestamp)
        timestamp = time.time()
    classwise_accuracy = np.divide(classwise_correct_num, classwise_num)
    print(classwise_accuracy)
    if log_writer is not None:
        log_writer.add_scalar('data/test_acc_s', classwise_accuracy[0], epoch_number)
        log_writer.add_scalar('data/test_acc_nos', classwise_accuracy[1], epoch_number)
        log_writer.add_scalar('data/test_loss', loss_meter.average, epoch_number)
    logging.warning(
        'Epoch: [{epoch}] -- TESTING SUMMARY\t'
        'Time {batch_time.sum:.2f}   '
        'Data {data_time.sum:.2f}   '
        'Loss {loss.average:.3f}     '
        'Top-1 {top1.average:.2f}    '.format(
            epoch=epoch_number, batch_time=batch_time_meter, data_time=data_time_meter,
            loss=loss_meter, top1=top1_meter))

    
def main(opt):
    opt.dataroot = opt.testroot
    opt.is_train = False
    test_loader, imgs = make_dataloader(opt, mode = 'val', print_fail_img = True)
    net = model_creator(opt)
    model_dict = torch.load(os.path.join(opt.experiments, 'checkpoint.pth.tar'))
    net = net.cuda()
    net = net.eval()
    #net.G = net.G.eval()
    net.load_state_dict(model_dict['state_dice'])
    eval_by_id(net, opt, test_loader, imgs)
if __name__ == '__main__':
    main(opt)
