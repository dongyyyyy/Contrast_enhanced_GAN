from train.train_CycleGAN import *

from train.train_pix2pix import *
from train.train_pix2pix_withPerceptualLoss import *
from train.withoutDiscriminator.train_sub_UNet import *
from train.train_pix2pix_withPA import *
if __name__ == '__main__':
    # training_pix2pix(size=[256,256],batch_size=128,gpu_num=[0,1,2,3])
    training_pix2pix_withPerceptualLoss_all(size=[256,256],batch_size=64,gpu_num=[0,1,2,3])
    # training_CycleGAN(size=[256,256],batch_size=32,gpu_num=[0,1,2,3])
    # img = cv2.imread('/home/eslab/dataset/Nect CT Png/enhanced_png/991619435/0126.png', cv2.IMREAD_COLOR)
    # print(img)
