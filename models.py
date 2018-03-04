import torch
import torchvision
from ops import stem, conv_block, define_G
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.models.squeezenet import SqueezeNet
from torchvision.models.alexnet import AlexNet
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision.models.densenet import DenseNet
model_urls = {'alexnet':'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'}
model_urls = {
    'alexnet':'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    }

def model_creator(opt):
    if opt.arch == 'squeeze_net':
        model = squeeze_net_dp(opt.num_classes)
    if opt.arch == 'resnet18':
        model = torchvision.models.resnet18(pretrained = False)
        model.fc = nn.Linear(512, opt.num_classes)
    if opt.arch == 'resnet9':
        model = resnet_dp(opt.num_classes)
    if opt.arch == 'resnet_multi_input':
        model = resnet18_multi_input(opt.num_classes)
    if opt.arch == 'sq_v1_mm':
        model = sq_v1_mm_dp(num_classes = opt.num_classes)
    if opt.arch == 'dense_dp':
        model = dense_net_dp(num_classes = opt.num_classes)   
    if opt.arch == 'alexnet':
        model = AlexNet()
        #model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
        model.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Linear(4096, opt.num_classes),
        )
    if opt.arch == 'alexnet_dp':
        model = Alexnet_dp(num_classes = opt.num_classes)
        pretrained_dict = model_zoo.load_url(model_urls['alexnet'])
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'conv' in k and 'G' not in k}
        #state = model.state_dict()
        #state.update(pretrained_dict)
        #model.load_state_dict(pretrained_dict)
        #model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    if opt.arch == 'resnet_quadruplets':
        model = resnet_quadruplets(opt.num_classes)
    return model
class Alexnet_dp(AlexNet):
    def __init__(self, num_classes):
        super(Alexnet_dp, self).__init__()
        self.G = define_G('experiments/latest_net_G_A.pth')
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes)
        )
        self.pool2d = nn.AvgPool2d(6)
    def forward(self, x):
        x1 = self.features(x)
        fake = self.G(x)
        x2 = self.features(fake)
        x = torch.cat([x1, x2], dim = 1)
        x = self.pool2d(x)
        x = x.view(x.size(0),512)
        x = self.classifier(x)
        return x
class feature_extract(ResNet):
    def __init__(self, block, layers, pretrained = True):
        super(feature_extract, self).__init__(block, layers)
        self.stem = stem(3,64)
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
class feature_extract_dp(DenseNet):
    def __init__(self, block_config = (1,1,1,1), num_classes = 4):
        self.num_classes = num_classes
        super(feature_extract_dp, self).__init__(block_config=block_config, num_classes = num_classes)
        self.final_conv = nn.Conv2d(512, self.num_classes, kernel_size = 1)
    def forward(self, x):
        x = self.features(x)
        #x = self.final_conv(x)
        return x
class sq_v1_mm_dp(nn.Module):
    def __init__(self, num_classes = 2):
        super(sq_v1_mm_dp, self).__init__()
        self.num_classes = num_classes
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(136, self.num_classes)
        self.modelA = feature_extract_dp(num_classes = num_classes)
        self.modelB = feature_extract_dp(num_classes = num_classes)
    def forward(self, x):
        x_split = torch.split(x, 3, dim = 1)
        x1 = x_split[0]
        x2 = x_split[1]
        o1 = self.modelA(x1)
        o2 = self.modelB(x2)
        o = torch.cat([o1, o2], dim= 1)
        o = self.avgpool(o)
        o = o.view(x.size(0), -1)
        o = self.fc(o)
        return o
class resnet18_multi_input(nn.Module):
    def __init__(self, num_classes = 8):
        super(resnet18_multi_input, self).__init__()
        self.num_classes = num_classes
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(1024, num_classes)
        self.modelA = feature_extract(BasicBlock, [1, 1, 1, 1])
        self.modelB = feature_extract(BasicBlock, [1, 1, 1, 1])
        self.G = define_G('experiments/latest_net_G_A.pth')
    def forward(self, x):
        x_fake = self.G(x)
        o1 = self.modelA(x)
        o2 = self.modelB(x_fake)
        o = torch.cat([o1, o2], dim = 1)
        o = self.avgpool(o)
        o = o.view(x.size(0), -1)
        o = self.fc(o)
        return o
class resnet_quadruplets(nn.Module):
    def __init__(self, num_classes = 8):
        super(resnet_quadruplets, self).__init__()
        self.num_classes = num_classes
        self.avgpool = nn.AvgPool2d(7)
        self.relu = nn.ReLU(inplace = True)
        self.feature_selction = nn.Linear(1024*2, 1024)
        self.feature_selction = conv_block(1024*2,1024,1,1,0)
        self.fc = nn.Linear(1024*2, num_classes)
        self.modelA = feature_extract(BasicBlock, [1, 1, 1, 1])
        self.modelB = feature_extract(BasicBlock, [1, 1, 1, 1])
        self.modelC = feature_extract(BasicBlock, [1,1,1,1])
        self.modelD = feature_extract(BasicBlock, [1, 1, 1,1])
    def forward(self, x):
        x_split = torch.split(x, 3, dim = 1)
        #print(x.size())
        x1 = x_split[0]
        x2 = x_split[1]
        x3 = x_split[2]
        x4 = x_split[3]
        o1 = self.modelA(x1)
        o2 = self.modelB(x2)
        o3 = self.modelC(x3)
        o4 = self.modelD(x4)
        o = torch.cat([o1,o2,o3,o4], dim = 1)
        #o = self.feature_selction(o)
        o = self.avgpool(o)
        o = o.view(x.size(0), -1)
        #o = self.feature_selction(o)
        #o = self.relu(o)
        o = self.fc(o)
        return o

class resnet_dp(nn.Module):
    def __init__(self, num_classes):
        super(resnet_dp, self).__init__()
        self.num_classes = num_classes
        self.feature_extractor = feature_extract(BasicBlock, [1,1,1,1])
        #self.feature_extractor.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
        pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in self.feature_extractor.state_dict()}
        #self.feature_extractor.load_state_dict(pretrained_dict)
        self.avgpool = nn.AvgPool2d(7, stride = 1)
        self.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        f = self.feature_extractor(x)
        out = self.avgpool(f)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
class dense_net_dp(nn.Module):
    def __init__(self, num_classes):
        super(dense_net_dp, self).__init__()
        self.num_classes = num_classes
        self.feature_extractor = feature_extract_dp(block_config=[2,2,2,2],num_classes = num_classes)
        self.avgpool = nn.AvgPool2d(7, stride = 1)
        self.relu = nn.ReLU(inplace = True)
        self.fc = nn.Linear(68, num_classes)
    def forward(self, x):
        f = self.feature_extractor(x)
        out = self.avgpool(f)
        out = self.relu(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
if __name__ == '__main__':
    from torch.autograd import Variable
    m = feature_extract(BasicBlock,  [1,1,1,1])
    pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
    pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in m.state_dict()}
    m.load_state_dict(pretrained_dict)
    x = Variable(torch.zeros((30,3,224,224)))
    x = m.forward(x)
    print(x.size())
