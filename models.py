import torch
import torchvision
import torch.nn as nn
def model_creator(opt):
    if opt.arch == 'squeeze_net':
        model = torchvision.models.squeezenet1_(pretrained = True)
    if opt.arch == 'resnet18':
        model = torchvision.models.resnet18(pretrained = True)
        model.fc = nn.Linear(512, opt.num_classes)
    return model

if __name__ == '__main__':
    from options import opt
    model_creator(opt)