import torch
import torchvision
from torchvision.models.resnet import ResNet, BasicBlock
import torch.nn as nn
def model_creator(opt):
    if opt.arch == 'squeeze_net':
        model = torchvision.models.squeezenet1_(pretrained = True)
    if opt.arch == 'resnet18':
        model = torchvision.models.resnet18(pretrained = False)
        model.fc = nn.Linear(512, opt.num_classes)
    if opt.arch == 'resnet18_multi_input':
        model = resnet18_multi_input()
    return model
class feature_extract(ResNet):
    def __init__(self, block, layers):
        super(feature_extract, self).__init__(block, layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
class resnet18_multi_input(nn.Module):
    def __init__(self, num_classes = 8):
        super(resnet18_multi_input, self).__init__()
        self.num_classes = num_classes
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(1024, num_classes)
        self.modelA = feature_extract(BasicBlock, [2, 2, 2, 2])
        self.modelB = feature_extract(BasicBlock, [2, 2, 2, 2])
    def forward(self, x):
        x_split = torch.split(x, 3, dim = 1)
        #print(x.size())
        x1 = x_split[0]
        x2 = x_split[1]
        o1 = self.modelA(x1)
        o2 = self.modelB(x2)
        o = torch.cat([o1, o2], dim = 1)
        o = self.avgpool(o)
        #print(o.size())
        o = o.view(x.size(0), -1)
        o = self.fc(o)
        return o

if __name__ == '__main__':
    from options import opt
    model_creator(opt)