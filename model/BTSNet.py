import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .SPP import ASPP_simple, ASPP
from .ResNet import ResNet101, ResNet18, ResNet34, ResNet50
from .resnet_aspp import ResNet_ASPP
import time


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

def softmax_2d(x):
    return torch.exp(x) / torch.sum(torch.sum(torch.exp(x), dim=-1, keepdim=True), dim=-2, keepdim=True)


class Bottleneck(nn.Module):
    
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=rate, padding=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.rate = rate

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BTSNet(nn.Module):
    def __init__(self, nInputChannels, n_classes, os, img_backbone_type='resnet50', depth_backbone_type='resnet50'):
        super(BTSNet, self).__init__()

        self.inplanes = 64
        self.os = os

        #ASPP模块空洞卷积rate
        if os == 16:
            aspp_rates = [1, 6, 12, 18]
        elif os == 8 or os == 32:
            aspp_rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        #os = output_stride
        if os == 16:
            strides = [1, 2, 2, 1]
            rates = [1, 1, 1, 2]
        elif os == 8:
            strides = [1, 2, 1, 1]
            rates = [1, 1, 2, 2]
        elif os == 32:
            strides = [1, 2, 2, 2]
            rates = [1, 1, 1, 1]
        else:
            raise NotImplementedError

        assert img_backbone_type == 'resnet50'

        # Modules
        self.conv1 = nn.Conv2d(nInputChannels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        layers = [3, 4, 6, 3]

        self.layer1 = self._make_layer( 64, layers[0], stride=strides[0], rate=rates[0])
        self.layer2 = self._make_layer( 128, layers[1], stride=strides[1], rate=rates[1])
        self.layer3 = self._make_layer( 256, layers[2], stride=strides[2], rate=rates[2])
        self.layer4 = self._make_layer( 512, layers[3], stride=strides[3], rate=rates[3])
        
        asppInputChannels = 2048
        asppOutputChannels = 256
        lowInputChannels =  256
        lowOutputChannels = 256

        self.aspp = ASPP(asppInputChannels, asppOutputChannels, aspp_rates)

        self.last_conv = nn.Sequential(
                nn.Conv2d(2*lowOutputChannels, lowOutputChannels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(lowOutputChannels),
                nn.ReLU(),
                nn.Conv2d(lowOutputChannels, lowOutputChannels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(lowOutputChannels),
                nn.ReLU(),
                nn.Conv2d(lowOutputChannels, n_classes, kernel_size=1, stride=1)
            )

        self.last_conv_rgb = nn.Sequential(
            nn.Conv2d(2*lowOutputChannels, lowOutputChannels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(lowOutputChannels),
            nn.ReLU(),
            nn.Conv2d(lowOutputChannels, lowOutputChannels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(lowOutputChannels),
            nn.ReLU(),
            nn.Conv2d(lowOutputChannels, n_classes, kernel_size=1, stride=1)
        )

        self.last_conv_depth = nn.Sequential(
            nn.Conv2d(2*lowOutputChannels, lowOutputChannels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(lowOutputChannels),
            nn.ReLU(),
            nn.Conv2d(lowOutputChannels, lowOutputChannels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(lowOutputChannels),
            nn.ReLU(),
            nn.Conv2d(lowOutputChannels, n_classes, kernel_size=1, stride=1)
        )

        # low_level_feature to 48 channels
        self.rgb_conv1_cp = BasicConv2d(64, lowOutputChannels,1)
        self.depth_conv1_cp = BasicConv2d(64, lowOutputChannels,1)
        self.rgb_layer1_cp = BasicConv2d(256, lowOutputChannels, 1)
        self.depth_layer1_cp = BasicConv2d(256, lowOutputChannels, 1)
        self.rgb_layer2_cp = BasicConv2d(512, lowOutputChannels, 1)
        self.depth_layer2_cp = BasicConv2d(512, lowOutputChannels, 1)
        self.rgb_layer3_cp = BasicConv2d(1024, lowOutputChannels, 1)
        self.depth_layer3_cp = BasicConv2d(1024, lowOutputChannels, 1)
        self.rgb_layer4_cp = BasicConv2d(2048, lowOutputChannels, 1)
        self.depth_layer4_cp = BasicConv2d(2048, lowOutputChannels, 1)

        self.fusion_high = BasicConv2d(2*lowOutputChannels,lowOutputChannels,3,1,1)
        self.fusion_low = BasicConv2d(2 * lowOutputChannels, lowOutputChannels, 3, 1, 1)
        self.fusion = BasicConv2d(2 * lowOutputChannels, lowOutputChannels, 3, 1, 1)



        self.resnet_aspp = ResNet_ASPP(nInputChannels, n_classes, os, depth_backbone_type)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv1_channel1 = nn.Conv2d(64, 64, 1, bias=True)
        self.conv1_spatial1 = nn.Conv2d(64, 1, 3, 1, 1,bias=True)

        self.layer1_channel1 = nn.Conv2d(256, 256, 1,bias=True)
        self.layer1_spatial1 = nn.Conv2d(256, 1, 3, 1, 1,bias=True)

        self.layer2_channel1 = nn.Conv2d(512, 512, 1,bias=True)
        self.layer2_spatial1 = nn.Conv2d(512, 1, 3, 1, 1,bias=True)

        self.layer3_channel1 = nn.Conv2d(1024, 1024, 1,bias=True)
        self.layer3_spatial1 = nn.Conv2d(1024, 1, 3, 1, 1,bias=True)

        self.layer4_channel1 = nn.Conv2d(2048, 2048, 1,bias=True)
        self.layer4_spatial1 = nn.Conv2d(2048, 1, 3, 1, 1,bias=True)

        self.conv1_channel2 = nn.Conv2d(64, 64, 1,bias=True)
        self.conv1_spatial2 = nn.Conv2d(64, 1, 3, 1, 1,bias=True)

        self.layer1_channel2 = nn.Conv2d(256, 256, 1,bias=True)
        self.layer1_spatial2 = nn.Conv2d(256, 1, 3, 1, 1,bias=True)

        self.layer2_channel2 = nn.Conv2d(512, 512, 1,bias=True)
        self.layer2_spatial2 = nn.Conv2d(512, 1, 3, 1, 1,bias=True)

        self.layer3_channel2 = nn.Conv2d(1024, 1024, 1,bias=True)
        self.layer3_spatial2 = nn.Conv2d(1024, 1, 3, 1, 1,bias=True)

        self.layer4_channel2 = nn.Conv2d(2048, 2048, 1,bias=True)
        self.layer4_spatial2 = nn.Conv2d(2048, 1, 3, 1, 1,bias=True)


    def _make_layer(self, planes, blocks, stride=1, rate=1):
        
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, rate, downsample))
        self.inplanes = planes * Bottleneck.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))

        return nn.Sequential(*layers)

    def bi_attention(self, img_feat, depth_feat, channel_conv1, spatial_conv1, channel_conv2, spatial_conv2):
        # spatial attention
        img_att = F.sigmoid(spatial_conv1(img_feat))
        depth_att = F.sigmoid(spatial_conv2(depth_feat))

        img_att = img_att + img_att * depth_att
        depth_att = depth_att + img_att * depth_att

        spatial_attentioned_img_feat = depth_att * img_feat
        spatial_attentioned_depth_feat = img_att * depth_feat

        # channel-wise attention
        img_vec = self.avg_pool(spatial_attentioned_img_feat)
        img_vec = channel_conv1(img_vec)
        img_vec = nn.Softmax(dim=1)(img_vec) * img_vec.shape[1]
        img_feat = spatial_attentioned_img_feat * img_vec

        depth_vec = self.avg_pool(spatial_attentioned_depth_feat)
        depth_vec = channel_conv2(depth_vec)
        depth_vec = nn.Softmax(dim=1)(depth_vec) * depth_vec.shape[1]
        depth_feat = spatial_attentioned_depth_feat * depth_vec

        return img_feat, depth_feat


    def forward(self, img, depth):
        x = self.conv1(img)
        x = self.bn1(x)
        x = self.relu(x)
        conv1_feat = x
        y = self.resnet_aspp.backbone_features.conv1(depth)
        y = self.resnet_aspp.backbone_features.bn1(y)
        y = self.resnet_aspp.backbone_features.relu(y)
        depth_conv1_feat = y
        x, y = self.bi_attention(x, y, self.conv1_channel1, self.conv1_spatial1, self.conv1_channel2,
                                 self.conv1_spatial2)

        after_depth_conv1_feat = y
        after_conv1_feat = x

        x = self.maxpool(x)
        x = self.layer1(x)
        layer1_feat = x
        y = self.resnet_aspp.backbone_features.maxpool(y)
        y = self.resnet_aspp.backbone_features.layer1(y)
        depth_layer1_feat = y
        x, y = self.bi_attention(x, depth_layer1_feat, self.layer1_channel1, self.layer1_spatial1, self.layer1_channel2,
                                 self.layer1_spatial2)

        after_layer1_feat = x
        after_depth_layer1_feat = y
        low_level_feature = x
        low_level_depth_feature = y

        x = self.layer2(x)
        layer2_feat = x
        y = self.resnet_aspp.backbone_features.layer2(y)
        depth_layer2_feat = y
        x, y = self.bi_attention(x, depth_layer2_feat, self.layer2_channel1, self.layer2_spatial1, self.layer2_channel2,
                                 self.layer2_spatial2)

        after_layer2_feat = x
        after_depth_layer2_feat =y

        x = self.layer3(x)
        layer3_feat = x
        y = self.resnet_aspp.backbone_features.layer3(y)
        depth_layer3_feat = y
        x, y = self.bi_attention(x, depth_layer3_feat, self.layer3_channel1, self.layer3_spatial1, self.layer3_channel2,
                                 self.layer3_spatial2)

        after_layer3_feat = x
        after_depth_layer3_feat = y

        x = self.layer4(x)
        layer4_feat = x
        y = self.resnet_aspp.backbone_features.layer4(y)
        depth_layer4_feat = y
        x, y = self.bi_attention(x, depth_layer4_feat, self.layer4_channel1, self.layer4_spatial1, self.layer4_channel2,
                                 self.layer4_spatial2)

        after_layer4_feat = x
        after_depth_layer4_feat = y

        if self.os == 32:
            x = F.upsample(x, scale_factor=4, mode='bilinear', align_corners=True)
            y = F.upsample(y, scale_factor=4, mode='bilinear', align_corners=True)
        
        x = self.aspp(x)
        x_aspp = x

        y = self.resnet_aspp.aspp(y)
        y_aspp = y


        after_layer4_feat = self.rgb_layer4_cp(after_layer4_feat)
        after_depth_layer4_feat = self.depth_layer4_cp(after_depth_layer4_feat)
        
        after_layer3_feat = self.rgb_layer3_cp(after_layer3_feat)
        after_depth_layer3_feat = self.depth_layer3_cp(after_depth_layer3_feat)
        after_layer2_feat = F.upsample(self.rgb_layer2_cp(after_layer2_feat), scale_factor=4, mode='bilinear',
                   align_corners=True)
        after_depth_layer2_feat = F.upsample(self.depth_layer2_cp(after_depth_layer2_feat), scale_factor=4, mode='bilinear',
                   align_corners=True)
        after_layer1_feat = F.upsample(self.rgb_layer1_cp(after_layer1_feat), scale_factor=2, mode='bilinear',
                   align_corners=True)
        after_depth_layer1_feat = F.upsample(self.depth_layer1_cp(after_depth_layer1_feat), scale_factor=2, mode='bilinear',
                   align_corners=True)
        after_conv1_feat = self.rgb_conv1_cp(after_conv1_feat)
        after_depth_conv1_feat = self.depth_conv1_cp(after_depth_conv1_feat)

        rgb_high_feat = x_aspp + after_layer4_feat +after_layer3_feat
        depth_high_feat = y_aspp + after_depth_layer4_feat + after_depth_layer3_feat
        rgb_low_feat = after_layer2_feat+after_layer1_feat+after_conv1_feat
        depth_low_feat = after_depth_layer2_feat+after_depth_layer1_feat+after_depth_conv1_feat

        #rgb_fusion
        rgb_feat = torch.cat((F.upsample(rgb_high_feat, scale_factor=8, mode='bilinear',
                   align_corners=True), rgb_low_feat), dim=1)
        sal_rgb = self.last_conv_rgb(rgb_feat)


        #depth_fusion
        depth_feat = torch.cat((F.upsample(depth_high_feat, scale_factor=8, mode='bilinear',
                   align_corners=True), depth_low_feat), dim=1)
        sal_depth = self.last_conv_depth(depth_feat)



        #final_fusion
        higt_feat = self.fusion_high(
            torch.cat((rgb_high_feat+depth_high_feat,rgb_high_feat*depth_high_feat), dim=1))
        low_feat = self.fusion_low(
            torch.cat((rgb_low_feat + depth_low_feat, rgb_low_feat * depth_low_feat), dim=1))
        
        high_feat = F.upsample(higt_feat, scale_factor=8, mode='bilinear',
                   align_corners=True)

        
        sal = self.last_conv(torch.cat((high_feat,low_feat),dim=1))
        sal = F.upsample(sal, img.size()[2:], mode='bilinear', align_corners=True)
        sal_rgb = F.upsample(sal_rgb, img.size()[2:], mode='bilinear', align_corners=True)
        sal_depth = F.upsample(sal_depth, img.size()[2:], mode='bilinear', align_corners=True)


        return sal, sal_rgb, sal_depth
