import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
import math

def weights_init(m, act_type='relu'):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if act_type == 'selu':
            n = float(m.in_channels * m.kernel_size[0] * m.kernel_size[1])
            m.weight.data.normal_(0.0, 1.0 / math.sqrt(n))
        else:
            m.weight.data.normal_(0.0, 0.02)        
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_scheduler(optimizer, args):
    if args.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = ((0.5 ** int(epoch >= 2)) *
                    (0.5 ** int(epoch >= 5)) *
                    (0.5 ** int(epoch >= 8)))
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=args.lr_decay_iters, gamma=0.1
        )
    elif args.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5
        )
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


class UnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_type='batch', act_type='selu'):
        super(UnetGenerator, self).__init__()
        self.name = 'unet'
        self.conv1 = nn.Conv2d(input_nc, ngf, 4, 2, 1)
        self.conv2 = nn.Conv2d(ngf, ngf * 2, 4, 2, 1)
        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1)
        self.conv4 = nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1)
        self.conv5 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.conv6 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.conv7 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.conv8 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.dconv1 = nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.dconv2 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1)
        self.dconv3 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1)
        self.dconv4 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1)
        self.dconv5 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, 4, 2, 1)
        self.dconv6 = nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, 4, 2, 1)
        self.dconv7 = nn.ConvTranspose2d(ngf * 2 * 2, ngf, 4, 2, 1)
        self.dconv8 = nn.ConvTranspose2d(ngf * 2, output_nc, 4, 2, 1)

        if norm_type == 'batch':
            self.norm = nn.BatchNorm2d(ngf)
            self.norm2 = nn.BatchNorm2d(ngf * 2)
            self.norm4 = nn.BatchNorm2d(ngf * 4)
            self.norm8 = nn.BatchNorm2d(ngf * 8)
        elif norm_type == 'instance':
            self.norm = nn.InstanceNorm2d(ngf)
            self.norm2 = nn.InstanceNorm2d(ngf * 2)
            self.norm4 = nn.InstanceNorm2d(ngf * 4)
            self.norm8 = nn.InstanceNorm2d(ngf * 8)
        self.leaky_relu = nn.LeakyReLU(0.2, True)

        if act_type == 'selu':
            self.act = nn.SELU(True)
        else:
            self.act = nn.ReLU(True)

        self.dropout = nn.Dropout(0.5)

        self.tanh = nn.Tanh()

    def forward(self, input):
        # Encoder
        # Convolution layers:
        # input is (nc) x 512 x 1024
        e1 = self.conv1(input)
        # state size is (ngf) x 256 x 512
        e2 = self.norm2(self.conv2(self.leaky_relu(e1)))
        # state size is (ngf x 2) x 128 x 256
        e3 = self.norm4(self.conv3(self.leaky_relu(e2)))
        # state size is (ngf x 4) x 64 x 128
        e4 = self.norm8(self.conv4(self.leaky_relu(e3)))
        # state size is (ngf x 8) x 32 x 64
        e5 = self.norm8(self.conv5(self.leaky_relu(e4)))
        # state size is (ngf x 8) x 16 x 32
        e6 = self.norm8(self.conv6(self.leaky_relu(e5)))
        # state size is (ngf x 8) x 8 x 16
        e7 = self.norm8(self.conv7(self.leaky_relu(e6)))
        # state size is (ngf x 8) x 4 x 8
        # No batch norm on output of Encoder
        e8 = self.conv8(self.leaky_relu(e7))

        # Decoder
        # Deconvolution layers:
        # state size is (ngf x 8) x 2 x 4
        d1_ = self.dropout(self.norm8(self.dconv1(self.act(e8))))
        # state size is (ngf x 8) x 4 x 8
        d1 = torch.cat((d1_, e7), 1)
        d2_ = self.dropout(self.norm8(self.dconv2(self.act(d1))))
        # state size is (ngf x 8) x 8 x 16
        d2 = torch.cat((d2_, e6), 1)
        d3_ = self.dropout(self.norm8(self.dconv3(self.act(d2))))
        # state size is (ngf x 8) x 16 x 32
        d3 = torch.cat((d3_, e5), 1)
        d4_ = self.norm8(self.dconv4(self.act(d3)))
        # state size is (ngf x 8) x 32 x 64
        d4 = torch.cat((d4_, e4), 1)
        d5_ = self.norm4(self.dconv5(self.act(d4)))
        # state size is (ngf x 4) x 64 x 128
        d5 = torch.cat((d5_, e3), 1)
        d6_ = self.norm2(self.dconv6(self.act(d5)))
        # state size is (ngf x 2) x 128 x 256
        d6 = torch.cat((d6_, e2), 1)
        d7_ = self.norm(self.dconv7(self.act(d6)))
        # state size is (ngf) x 256 x 512
        d7 = torch.cat((d7_, e1), 1)
        d8 = self.dconv8(self.act(d7))
        # state size is (nc) x 512 x 1024
        output = self.tanh(d8)
        return output


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_type='instance', act_type='selu', use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()

        self.name = 'resnet'
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        use_bias = norm_type == 'instance'

        if norm_type == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d

        if act_type == 'selu':
            self.act = nn.SELU(True)
        else:
            self.act = nn.ReLU(True)

        
        self.reflect1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias)
        self.norm1 = norm_layer(ngf)
        self.act1 = self.act
        self.conv2 = nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=use_bias)
        self.norm2 = norm_layer(ngf * 2)
        self.act2 = self.act
        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=use_bias)
        self.norm3 = norm_layer(ngf * 4)
        self.act3 = self.act

        self.res1 = ResnetBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.res2 = ResnetBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.res3 = ResnetBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.res4 = ResnetBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.res5 = ResnetBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.res6 = ResnetBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)


        self.dconv1 = nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias)
        self.dnorm1 = norm_layer(ngf * 2)
        self.dact1 = self.act
        self.dconv2 = nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias)
        self.dnorm2 = norm_layer(ngf)
        self.dact2 = self.act

        self.reflect2 = nn.ReflectionPad2d(3)
        self.conv4 = nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)
        self.tanh = nn.Tanh() 


    def forward(self, input):
        input = input
        enc = self.act3(self.norm3(self.conv3(self.act2(self.norm2(self.conv2(self.act1(self.norm1(self.conv1(self.reflect1(input))))))))))
        res = self.res6(self.res5(self.res4(self.res3(self.res2(self.res1(enc))))))
        dec = self.dact2(self.dnorm2(self.dconv2(self.dact1(self.dnorm1(self.dconv1(res))))))
        output = self.tanh(self.conv4(self.reflect2(dec)))
        return output
        


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



