import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F


class CycleGAN_D(nn.Module):
    def __init__(self,input_channels=1,norm_layer='instance',use_bias=False):
        super().__init__()
        self.use_bias = False
        self.conv_input = nn.Conv2d(input_channels,64,4,stride=2,padding=1)
        self.leakyReLU = nn.LeakyReLU(0.2,inplace=True)

        self.block1 = self.make_blocks(in_channels=64, out_channels=128,kernel_size=4,stride=2,padding=1,norm_layer=norm_layer)
        self.block2 = self.make_blocks(in_channels=128, out_channels=256, kernel_size=4,stride=2,padding=1,norm_layer=norm_layer)
        self.block3 = self.make_blocks(in_channels=256,out_channels=512,kernel_size=4,stride=1,padding=1,norm_layer=norm_layer)

        self.conv_output = nn.Conv2d(512,1,4,padding=1)



    def make_blocks(self,in_channels,out_channels,kernel_size,stride,padding=1,norm_layer='instance'):
        blocks = []
        blocks += [nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding)]
        if norm_layer == 'instance':
            blocks += [nn.InstanceNorm2d(out_channels)]
        elif norm_layer =='batch':
            blocks += [nn.BatchNorm2d(out_channels)]
        blocks += [nn.LeakyReLU(0.2,inplace=True)]

        return nn.Sequential(*blocks)

    def forward(self,x):
        out = self.conv_input(x)
        out = self.leakyReLU(out)

        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)

        out = self.conv_output(out) # [1,26,26]

        out = F.avg_pool2d(out,out.size()[2:]).view(out.size()[0],-1) # output = 1로 압축
        # print(out.shape)
        return out

# summary(CycleGAN_D().cuda(),(1,224,224))