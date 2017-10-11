import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required = True, help = 'cifar10|place365|imagenet|glomerulus')
parser.add_argument('--dataroot', required = True,help = 'ptah to raw images')

parser.add_argement('--arch', required = False, help = 'resnet18|cifar-net|' )
parser.add_argument('--epochs', default = 90, type = int, help = 'epoch to run')
parser.add_argument('--batch_size', default = 64, type = int, help = 'batch size')
parser.add_argument('--num_workers', default = 1, type = int, help = 'num of workers')
parser.add_argument('--lr', default = 0.01, type = float, help = 'initial learning rate')
