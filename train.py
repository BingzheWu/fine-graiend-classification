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
from eval import eval
def train(opt):
    data = make_dataset(opt)
    testLoader = data
    net = model_creator(opt)
    if opt.use_cuda:
        print("load cuda model")
        #net = torch.nn.DataParallel(net).cuda()
        net = net.cuda()
    optimizer = optim.SGD(net.parameters(), lr = opt.lr, momentum = 0.9)
    for epoch in range(opt.epochs):
        net.train()
        for batch_idx, (image, target) in enumerate(data):
            image = Variable(image)
            target = Variable(target)
            if opt.use_cuda:
                image = image.cuda()
                target = target.cuda()
            logits = net(image)
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 50 == 0:
                print(loss.data[0])
        print("start validate")
        eval(net,opt, testLoader)
        torch.save(net.state_dict(), os.path.join(opt.experiments, 'model_{0}'.format(epoch)))
        
if __name__ == '__main__':
    train(opt)
