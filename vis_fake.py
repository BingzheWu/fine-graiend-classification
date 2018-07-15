from ops import define_G
import torch
import scipy.misc
import numpy as np
import sys
from torch.autograd import Variable
def vis(source_image_file, target_file):
    source_image = scipy.misc.imread(source_image_file)
    source_image = scipy.misc.imresize(source_image, (224, 224))
    print(source_image.shape)
    source_image = np.transpose(source_image, (2,0,1))
    source_image = np.expand_dims(source_image, 0)
    print(source_image.shape)
    G_pasm = define_G('experiments/pas_he.pth')
    #G_pasm = G_pasm.eval()
    img_tensor = torch.Tensor(source_image)
    #img_tensor = img_tensor.view(1, 3, 224,224)
    #scipy.misc.imsave('resize_test.jpg', img_tensor.view(224,224,3).numpy())
    pasm_fake = G_pasm(Variable(img_tensor))
    #pasm_fake = pasm_fake.transpose(0,2,3,1)
    pasm_fake = pasm_fake.squeeze()
    pasm_fake = pasm_fake.data.float().numpy()
    print(pasm_fake.shape)
    pasm_fake = np.transpose(pasm_fake, (1,2,0))
    #pasm_fake = (np.transpose(pasm_fake, (1, 2, 0)) + 1) / 2.0 * 255.0
    scipy.misc.imsave(target_file, pasm_fake)

if __name__ == '__main__':
    source = sys.argv[1]
    target = sys.argv[2]
    vis(source, target)
