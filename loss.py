import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ops import onehot
class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha = None, gamma = 1, size_average = True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)
        class_mask = onehot(targets.view(-1).data, self.class_num)
        class_mask = Variable(class_mask)
        if inputs.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[targets.view(-1, 1).data.view(-1)]
        probs = (P*class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss