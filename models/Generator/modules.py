import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self,in_features,norm_layer='instance',kernel_size=3,dropout_p=0.,use_bias=False,padding_type ='reflect'):
        super(ResidualBlock,self).__init__()
        self.conv = self.make_blocks(kernel_size=kernel_size,in_features=in_features,padding_type=padding_type,norm_layer=norm_layer,dropout_p=dropout_p,use_bias=use_bias)


    def make_blocks(self,kernel_size=3,in_features=256,padding_type='reflect',norm_layer='instance',dropout_p=0.,use_bias=False):
        conv_block = []
        conv_p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type =='replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        else:
            conv_p = 1

        conv_block += [nn.Conv2d(in_channels=in_features,out_channels=in_features,kernel_size=kernel_size,padding=conv_p,bias=use_bias)]
        if norm_layer == 'instance':
            conv_block += [nn.InstanceNorm2d(num_features=in_features,affine=False,track_running_stats=False)]
        elif norm_layer == 'batch':
            conv_block += [nn.BatchNorm2d(num_features=in_features,affine=True,track_running_stats=True)]
        conv_block += [nn.ReLU(inplace=True)]
        if dropout_p > 0.:
            conv_block += [nn.Dropout(dropout_p)]

        conv_p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        else:
            conv_p = 1

        conv_block += [
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size, padding=conv_p,
                      bias=use_bias)]
        if norm_layer == 'instance':
            conv_block += [nn.InstanceNorm2d(num_features=in_features, affine=False, track_running_stats=False)]
        elif norm_layer == 'batch':
            conv_block += [nn.BatchNorm2d(num_features=in_features, affine=True, track_running_stats=True)]

        return nn.Sequential(*conv_block)

    def forward(self,x):
        return self.conv(x)