import torch
import torch.nn as nn
import functools

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
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

def get_norm_layer(norm_type):
    norm_layer = functools.partial(nn.InstanceNorm2d, affine = False)
    return norm_layer
def define_G(g_pt_dir=None, input_nc=3, output_nc=3, ngf=64, norm = 'instance', use_dropout = True, init_type = 'normal', gpu_ids = []):
    norm_layer = get_norm_layer(norm_type = norm)
    netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer = norm_layer, use_dropout = use_dropout, n_blocks = 9, gpu_ids = gpu_ids)
    #netG.load_state_dict(torch.load(g_pt_dir))
    return netG