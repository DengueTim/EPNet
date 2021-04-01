"""
implementation of the PWC-DC network for optical flow estimation by Sun et al., 2018

Jinwei Gu and Zhile Ren

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import numpy as np
os.environ['PYTHON_EGG_CACHE'] = 'tmp/' # a writable directory 


from models.FlowNetUtil import multiscaleEPE


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):   
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                        padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1))


def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(int(in_planes), out_planes, kernel_size, stride, padding, bias=True)

# Pyramidal processing, Warping, Simple disparity

class FlowNetPwsG(nn.Module):
    """
    PWC-DC net. add dilation convolution and densenet connections

    """
    def __init__(self, channels=[32,128,1024]):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warpping

        """
        super(FlowNetPwsG,self).__init__()

        self.flowEncoder1 = nn.Sequential(
            nn.Conv2d(2, channels[0], kernel_size=(3, 5)),
            nn.BatchNorm2d(channels[0]),
            nn.LeakyReLU(),
            nn.Conv2d(channels[0], channels[1], kernel_size=(3, 5)),
            nn.BatchNorm2d(channels[1]),
            nn.LeakyReLU(),
            nn.Conv2d(channels[1], channels[2], kernel_size=(3, 5)),
            nn.BatchNorm2d(channels[2]),
            nn.LeakyReLU()
        )

        self.flowDecoder1 = nn.Sequential(
            nn.ConvTranspose2d(channels[2], channels[1], kernel_size=(3, 5)),
            nn.BatchNorm2d(channels[1]),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(channels[1], channels[0], kernel_size=(3, 5)),
            nn.BatchNorm2d(channels[0]),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(channels[0], 2, kernel_size=(3, 5)),
        )

    def forward(self,im17, im27):
        im16 = F.interpolate(im17, (256, 448), mode='area')
        im26 = F.interpolate(im27, (256, 448), mode='area')
        im15 = F.interpolate(im16, (128, 224), mode='area')
        im25 = F.interpolate(im26, (128, 224), mode='area')
        im14 = F.interpolate(im15, (64, 112), mode='area')
        im24 = F.interpolate(im25, (64, 112), mode='area')
        im13 = F.interpolate(im14, (32, 56), mode='area')
        im23 = F.interpolate(im24, (32, 56), mode='area')
        im12 = F.interpolate(im13, (16, 28), mode='area')
        im22 = F.interpolate(im23, (16, 28), mode='area')
        im11 = F.interpolate(im12, (8, 14), mode='area')
        im21 = F.interpolate(im22, (8, 14), mode='area')

        nn.AvgPool2d()

        flow_encoded_1 = self.flowEncoder1(torch.cat((im11,im21),1))
        flow = self.flowDecoder1(flow_encoded_1)

        return [flow]

        # flow0 = F.interpolate(flow2, scale_factor=4, mode='bilinear')
        # return flow0

    @staticmethod
    def loss(prediction, target):
        return multiscaleEPE(prediction, target, weights=[1.0])