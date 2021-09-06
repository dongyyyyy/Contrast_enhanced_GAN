from models.Generator.modules import *
import torch.nn as nn
from torchsummary import summary

class CycleGAN_G(nn.Module):
    def __init__(self, input_channels=1,features=64,output_channels=1,norm_layer='instance',kernel_size=3,dropout_p=0.,use_bias=False,padding_type ='reflect', n_residual_blocks=9):
        super().__init__()
        # be used params for Generator
        self.input_channels = input_channels
        self.features = features
        self.output_channels = output_channels
        self.norm_layer = norm_layer
        self.dropout_p = dropout_p
        self.use_bias = use_bias
        self.padding_type = padding_type


        self.conv_input = self.make_inputLayer()
        self.input_channel = features
        self.output_channel = features * 2
        self.downsample_block = self.make_downsampleLayer()

        self.residual_block = self.make_residualBlocks(kernel_size=kernel_size,n_residual_blocks=n_residual_blocks)
        self.upsample_block = self.make_upsampleLayer()

        self.conv_output = self.make_outputLayer()

    def make_inputLayer(self):
        blocks = []
        p = 0
        if self.padding_type == 'reflect':
            blocks += [nn.ReflectionPad2d(3)]
        elif self.padding_type == 'replicate':
            blocks += [nn.ReplicationPad2d(3)]
        else:
            p = 3
        blocks += [nn.Conv2d(in_channels=self.input_channels, out_channels=self.features, kernel_size=7, stride=1, padding=p,
                                 bias=self.use_bias)]
        if self.norm_layer == 'instance':
            blocks += [nn.InstanceNorm2d(num_features=self.features, affine=False, track_running_stats=False)]
        elif self.norm_layer == 'batch':
            blocks += [nn.BatchNorm2d(num_features=self.features, affine=True, track_running_stats=True)]
        blocks += [nn.ReLU(inplace=True)]

        return nn.Sequential(*blocks)

    def make_downsampleLayer(self):
        blocks = []
        for _ in range(2):
            blocks += [nn.Conv2d(self.input_channel, self.output_channel, kernel_size=3, stride=2, padding=1,bias=self.use_bias)]
            if self.norm_layer == 'instance':
                blocks += [nn.InstanceNorm2d(num_features=self.output_channel, affine=False, track_running_stats=False)]
            elif self.norm_layer == 'batch':
                blocks += [nn.BatchNorm2d(num_features=self.output_channel, affine=True, track_running_stats=True)]
            blocks += [nn.ReLU(inplace=True)]
            self.input_channel = self.output_channel
            self.output_channel = self.input_channel * 2

        return nn.Sequential(*blocks)

    def make_residualBlocks(self,kernel_size=3,n_residual_blocks=9):
        blocks = []
        print('self.input_channel : ',self.input_channel)
        for _ in range(n_residual_blocks):
            blocks += [ResidualBlock(in_features=self.input_channel, norm_layer=self.norm_layer, kernel_size=kernel_size, dropout_p=self.dropout_p, use_bias=self.use_bias,
                        padding_type=self.padding_type)]
        return nn.Sequential(*blocks)

    def make_upsampleLayer(self):
        blocks = []
        self.output_channel = self.input_channel // 2
        for _ in range(2):
            blocks += [nn.ConvTranspose2d(self.input_channel,self.output_channel, kernel_size=3, stride=2, padding=1,output_padding=1,bias=self.use_bias)]
            if self.norm_layer == 'instance':
                blocks += [nn.InstanceNorm2d(num_features=self.output_channel, affine=False, track_running_stats=False)]
            elif self.norm_layer == 'batch':
                blocks += [nn.BatchNorm2d(num_features=self.output_channel, affine=True, track_running_stats=True)]
            blocks += [nn.ReLU(inplace=True)]
            self.input_channel = self.output_channel
            self.output_channel = self.input_channel // 2

        return nn.Sequential(*blocks)

    def make_outputLayer(self):
        blocks = []
        p = 0
        if self.padding_type == 'reflect':
            blocks += [nn.ReflectionPad2d(3)]
        elif self.padding_type == 'replicate':
            blocks += [nn.ReplicationPad2d(3)]
        else:
            p = 3
        blocks += [nn.Conv2d(self.input_channel,self.output_channels,kernel_size=7,stride=1,padding=p,bias=self.use_bias)]
        blocks += [nn.Tanh()]

        return nn.Sequential(*blocks)

    def forward(self,x):
        out = self.conv_input(x)
        out = self.downsample_block(out)
        # print(out.shape)
        out = self.residual_block(out)
        out = self.upsample_block(out)
        out = self.conv_output(out)
        return out
