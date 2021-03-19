import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
from .FlowNetUtil import conv, predict_flow, deconv, crop_like

__all__ = [
    'flownets', 'flownets_bn'
]


class FlowNetS(nn.Module):
    expansion = 1

    def __init__(self,batchNorm=True):
        super(FlowNetS,self).__init__()

        self.batchNorm = batchNorm
        self.conv1   = conv(self.batchNorm,   2,   24, kernel_size=7, stride=2)
        self.conv2   = conv(self.batchNorm,  24,  48, kernel_size=5, stride=2)
        self.conv3   = conv(self.batchNorm, 48,  96, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 96,  96)
        self.conv4   = conv(self.batchNorm, 96,  128, stride=2)
        self.conv4_1 = conv(self.batchNorm, 128,  128)

        # self.conv5   = conv(self.batchNorm, 512,  512, stride=2)
        # self.conv5_1 = conv(self.batchNorm, 512,  512)
        # self.conv6   = conv(self.batchNorm, 512, 1024, stride=2)
        # self.conv6_1 = conv(self.batchNorm,1024, 1024)

        # self.deconv5 = deconv(1024,512)
        # self.deconv4 = deconv(1026,256)
        self.deconv3 = deconv(128,32)
        self.deconv2 = deconv(130,24)
        self.deconv1 = deconv(74, 16)

        # self.predict_flow6 = predict_flow(1024)
        # self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(128)
        self.predict_flow3 = predict_flow(130)
        self.predict_flow2 = predict_flow(74)
        self.predict_flow1 = predict_flow(42)

        # self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        # self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow1_to_0 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        # out_conv5 = self.conv5_1(self.conv5(out_conv4))
        # out_conv6 = self.conv6_1(self.conv6(out_conv5))

        # flow6       = self.predict_flow6(out_conv6)
        # flow6_up    = crop_like(self.upsampled_flow6_to_5(flow6), out_conv5)
        # out_deconv5 = crop_like(self.deconv5(out_conv6), out_conv5)
        #
        # concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
        # flow5       = self.predict_flow5(concat5)
        # flow5_up    = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)
        # out_deconv4 = crop_like(self.deconv4(concat5), out_conv4)
        #
        # concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        flow4       = self.predict_flow4(out_conv4)
        flow4_up    = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)
        out_deconv3 = crop_like(self.deconv3(out_conv4), out_conv3)

        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        flow3       = self.predict_flow3(concat3)
        flow3_up    = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2)
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2)

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        flow2 = self.predict_flow2(concat2)
        flow2_up = crop_like(self.upsampled_flow2_to_1(flow2), out_conv1)
        out_deconv1 = crop_like(self.deconv1(concat2), out_conv1)

        concat1 = torch.cat((out_conv1, out_deconv1, flow2_up), 1)
        flow1 = self.predict_flow1(concat1)

        flow1_up = self.upsampled_flow1_to_0(flow1)


        # if self.training:
        #     return flow2,flow3,flow4,flow5,flow6
        # else:
        #     return flow2
        return flow1_up[:, :, 2:-2, 2:-2]  # Original computes loss over scaled flows but heavily weighted to the highest res flow.

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def flownets(data=None):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = FlowNetS(batchNorm=False)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model


def flownets_bn(data=None):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = FlowNetS(batchNorm=True)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model
