import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
#from resnext import ResNeXt101
from torch.autograd import Variable
from .model2_graph4_hrnet_agcm import Model
model6 = Model()
model6.cuda()
class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.cosalnet = model6
        self.jpu = JPU([96, 96, 96, 192, 384], width=96, norm_layer=nn.BatchNorm2d)
        de_in_channels=int(96*4)
        de_layers = make_decoder_layers(decoder_archs['d16'], de_in_channels, batch_norm=True)
        self.decodersal = DOCSDecoderNet(de_layers)
#        self.initialize_weights()
    def forward(self, img):  
        cosalmap, fea3, fea2, fea1 = self.cosalnet(img)
        with torch.no_grad():
          feat= self.jpu(fea1, fea2, fea3)
          pred=self.decodersal(feat)
          pred=F.upsample(pred, size=img.size()[2:], mode='bilinear')
        #cosalmap=F.upsample(cosalmap, img.size()[2:], mode='bilinear')
        return pred, cosalmap
    
#    def initialize_weights(self):
#        pretrained_dict=torch.load('/home/litengpeng/CODE/cosal/aaai19/mine1-cosal/model_path/graph4_decoder_hrnet/hrnet_iter67500.pth')
#        net_dict=self.cosalnet.state_dict()
#        for key,value in pretrained_dict.items():
#          if key in net_dict.keys():
#             net_dict[key]=value
#        net_dict.update(net_dict)
#        self.cosalnet.load_state_dict(net_dict)    
        
class JPU(nn.Module):
    def __init__(self, in_channels, width=96, norm_layer=nn.BatchNorm2d):
        super(JPU, self).__init__()
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))
        
        self.dilation1 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=1, dilation=1, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=2, dilation=2, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=4, dilation=4, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation4 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=8, dilation=8, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))

    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.upsample(feats[-2], (h, w), mode='bilinear')
        feats[-3] = F.upsample(feats[-3], (h, w), mode='bilinear')
        feat = torch.cat(feats, dim=1)   #### channel 128*3=284
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)], dim=1)

        return feat  ##### channel 128*4=512

class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x
#################
decoder_archs = {
    'd16': [96*4, 'd256', 256, 256, 'd128', 128, 128, 'd64', 64, 64, 'c1']
}

def make_decoder_layers(cfg, in_channels, batch_norm=True):
    layers = []
    for v in cfg:
        if type(v) is str:
            if v[0] == 'd':
                v = int(v[1:])
                convtrans2d = nn.ConvTranspose2d(in_channels, v, kernel_size=4, stride=2, padding=1)
                if batch_norm:
                    layers += [convtrans2d, nn.BatchNorm2d(v), nn.ReLU()]
                else:
                    layers += [convtrans2d, nn.ReLU()]
                in_channels = v
            elif v[0] == 'c':
                v = int(v[1:])
                layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.Sequential(*layers)
class DOCSDecoderNet(nn.Module):
    def __init__(self, features):
        super(DOCSDecoderNet, self).__init__()
        self.features = features

    def forward(self, x):
        return self.features(x)