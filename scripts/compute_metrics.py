import numpy as np

import sys
noa_num = 2828
s_num = 237 + 618

def compute_f1(s_accuracy, noa_accuracy):
    tp = s_accuracy * s_num
    fn = s_num - tp
    tn = noa_accuracy * noa_num
    fp = noa_num - tn
    f1 = 2.0*tp / float(2*tp + fp + fn)
    print("f1:{}.3f".format(f1))
if __name__ == '__main__':
    s_acc = float(sys.argv[1])
    noa_acc = float(sys.argv[2])
    compute_f1(s_acc, noa_acc)
