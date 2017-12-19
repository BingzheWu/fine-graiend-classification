import torch
import torch.nn as nn


def conv_block(in_c, out_c, k_size, strides, padding, name='conv_blcok', 
	alpha = 0., bias = False, batch_norm = True):
	out = nn.Sequential()
	out.add_module(name+'_conv', nn.Conv2d(in_c, out_c, k_size, strides, padding, bias = bias))
	if batch_norm:
		out.add_module(name+'_norm', nn.BatchNorm2d(out_c))
	out.add_module(name+'_activation', nn.LeakyReLU(alpha, inplace = True))
	return out
def upsample(in_c, out_c, k_size, strides, padding, name, alpha = 0.2, 
	bias = False, batch_norm = False):
	out = nn.Sequential()
	out.add_module(name+'.conv', nn.ConvTranspose2d(in_c, out_c, k_size, strides, padding, bias = bias))
	if batch_norm:
		out.add_module(name+'.norm', nn.BatchNorm2d(out_c))
	out.add_module(name+'.activation', nn.LeakyReLU(alpha, inplace = True))
	return out
def stem(in_c, out_c, name = 'stem'):
	out = nn.Sequential()
	out.add_module(name+'conv1', conv_block(in_c, out_c, 3, padding = 1, strides = 2))
	out.add_module(name+'conv2', conv_block(out_c, 2*out_c, 3, padding = 1, strides = 1))
	out.add_module(name+'conv3', conv_block(2*out_c, out_c, 3, padding = 1, strides = 1))
	out.add_module(name+'pool', nn.MaxPool2d(2,2))
	return out
def onehot(x, num_classes):
    ones = torch.sparse.torch.eye(num_classes)
    ones = ones.cuda()
    return ones.index_select(0, x)