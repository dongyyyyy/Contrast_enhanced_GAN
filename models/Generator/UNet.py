from models.Generator.modules import *
import torch.nn as nn
from torchsummary import summary
import torch

class UNet_block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3,norm_layer='instance',middle_channel=None,use_bias=False,activation_func = 'ReLU'):
        super().__init__()
        if middle_channel == None:
            middle_channel = out_channel
            self.blocks = self.makeBlocks(in_channel=in_channel,middle_channel=middle_channel,out_channel=out_channel,
                                        kernel_size=kernel_size,norm_layer=norm_layer,activation_func=activation_func,use_bias=use_bias)

    def makeBlocks(self,in_channel,middle_channel,out_channel,kernel_size,norm_layer,use_bias,activation_func):
        blocks = []

        blocks += [nn.Conv2d(in_channels=in_channel,out_channels=middle_channel,kernel_size = kernel_size,stride=1,padding=kernel_size//2,bias=use_bias)]
        if norm_layer == 'instance':
            blocks += [nn.InstanceNorm2d(num_features=middle_channel, affine=False, track_running_stats=False)]
        elif norm_layer == 'batch':
            blocks += [nn.BatchNorm2d(num_features=middle_channel, affine=True, track_running_stats=True)]
        if activation_func =='ReLU':
            blocks += [nn.ReLU(inplace=True)]
        elif activation_func =='leakyReLU':
            blocks += [nn.LeakyReLU(negative_slope=0.01,inplace=True)]

        blocks += [nn.Conv2d(in_channels=middle_channel,out_channels=out_channel,kernel_size = kernel_size,stride=1,padding=kernel_size//2,bias=use_bias)]
        if norm_layer == 'instance':
            blocks += [nn.InstanceNorm2d(num_features=out_channel, affine=False, track_running_stats=False)]
        elif norm_layer == 'batch':
            blocks += [nn.BatchNorm2d(num_features=out_channel, affine=True, track_running_stats=True)]
        if activation_func =='ReLU':
            blocks += [nn.ReLU(inplace=True)]
        elif activation_func =='leakyReLU':
            blocks += [nn.LeakyReLU(negative_slope=0.01,inplace=True)]
        
        return nn.Sequential(*blocks)

    def forward(self,x):
        return self.blocks(x)


class Unet_downBlock(nn.Module):
    def __init__(self,in_channel, out_channel, kernel_size=3,norm_layer='instance',middle_channel=None,use_bias=False,activation_func = 'ReLU',scale=2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=scale),
            UNet_block(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size,norm_layer=norm_layer,
                        middle_channel=middle_channel,use_bias=use_bias,activation_func = activation_func))
    def forward(self, x):
        return self.maxpool_conv(x)

class UNet_upBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, norm_layer='instance',middle_channel=None,use_bias=False,activation_func='ReLU',scale=2,bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
            self.conv = UNet_block(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size,norm_layer=norm_layer,
                        middle_channel=in_channel//2,use_bias=use_bias,activation_func = activation_func)
        else:
            self.up = nn.ConvTranspose2d(in_channel, in_channel // 2, kernel_size=2, stride=2,bias=use_bias)
            self.conv = UNet_block(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size,norm_layer=norm_layer,
                        middle_channel=middle_channel,use_bias=use_bias,activation_func = activation_func)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2,x1],dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1,out_channels=1,kernel_size=3,norm_layer='batch', activation_func='ReLU',use_bias=False,bilinear=False,scale=2):
        super().__init__()
        self.in_conv = UNet_block(in_channel=in_channels, out_channel=64,kernel_size=kernel_size,norm_layer=norm_layer,middle_channel=None,use_bias=use_bias,activation_func=activation_func)
        # in_channel, out_channel, kernel_size=3,norm_layer='instance',middle_channel=None,use_bias=False,activation_func = 'ReLU'
        self.down1 = Unet_downBlock(in_channel=64, out_channel=128,kernel_size=kernel_size,norm_layer=norm_layer,middle_channel=None,
                                    use_bias=use_bias,activation_func=activation_func,scale=scale)
        self.down2 = Unet_downBlock(in_channel=128, out_channel=256,kernel_size=kernel_size,norm_layer=norm_layer,middle_channel=None,
                                    use_bias=use_bias,activation_func=activation_func,scale=scale)
        self.down3 = Unet_downBlock(in_channel=256, out_channel=512,kernel_size=kernel_size,norm_layer=norm_layer,middle_channel=None,
                                    use_bias=use_bias,activation_func=activation_func,scale=scale)
        self.down4 = Unet_downBlock(in_channel=512, out_channel=1024,kernel_size=kernel_size,norm_layer=norm_layer,middle_channel=None,
                                    use_bias=use_bias,activation_func=activation_func,scale=scale)

        # in_channel, out_channel, kernel_size=3, norm_layer='instance',middle_channel=None,use_bias=False,activation_func='ReLU',scale=2,bilinear=False
        self.up1 = UNet_upBlock(in_channel=1024, out_channel=512, kernel_size=3, norm_layer=norm_layer,middle_channel=None,
                                use_bias=use_bias,activation_func=activation_func,scale=scale,bilinear=bilinear)
        self.up2 = UNet_upBlock(in_channel=512, out_channel=256, kernel_size=3, norm_layer=norm_layer,middle_channel=None,
                                use_bias=use_bias,activation_func=activation_func,scale=scale,bilinear=bilinear)
        self.up3 = UNet_upBlock(in_channel=256, out_channel=128, kernel_size=3, norm_layer=norm_layer,middle_channel=None,
                                use_bias=use_bias,activation_func=activation_func,scale=scale,bilinear=bilinear)
        self.up4 = UNet_upBlock(in_channel=128, out_channel=64, kernel_size=3, norm_layer=norm_layer,middle_channel=None,
                                use_bias=use_bias,activation_func=activation_func,scale=scale,bilinear=bilinear)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1,bias=use_bias)
 

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return logits