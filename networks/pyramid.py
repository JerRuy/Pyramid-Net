import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from functools import partial

nonlinearity = partial(F.relu, inplace=True)


class SpatialAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttentionBlock, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=(1, 3), padding=(0, 1))
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=(3, 1), padding=(1, 0))
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        # compress x: [B,C,H,W]-->[B,H*W,C], make a matrix transpose
        proj_query = self.query(x).view(B, -1, W * H).permute(0, 2, 1)
        proj_key = self.key(x).view(B, -1, W * H)
        affinity = torch.matmul(proj_query, proj_key)
        affinity = self.softmax(affinity)
        proj_value = self.value(x).view(B, -1, H * W)
        weights = torch.matmul(proj_value, affinity.permute(0, 2, 1))
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights + x
        return out


class ChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionBlock, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        proj_query = x.view(B, C, -1)
        proj_key = x.view(B, C, -1).permute(0, 2, 1)
        affinity = torch.matmul(proj_query, proj_key)
        affinity_new = torch.max(affinity, -1, keepdim=True)[0].expand_as(affinity) - affinity
        affinity_new = self.softmax(affinity_new)
        proj_value = x.view(B, C, -1)
        weights = torch.matmul(affinity_new, proj_value)
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights + x
        return out


class attention(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, ch):
        super(attention, self).__init__()
        self.sab = SpatialAttentionBlock(ch)
        self.cab = ChannelAttentionBlock(ch)

        self.br = nn.Sequential(
            nn.Conv2d(ch, ch, 1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        sab = self.sab(x)
        cab = self.cab(x)
        x = sab + cab
        return x


# class PSPModule(nn.Module):
#     def __init__(self, features, out_features=1024, sizes=(2, 3, 6, 14)):
#         super().__init__()
#         self.stages = []
#         self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
#         self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
#         self.relu = nn.ReLU()
#
#     def _make_stage(self, features, size):
#         prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
#         conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
#         return nn.Sequential(prior, conv)
#
#     def forward(self, feats):
#         h, w = feats.size(2), feats.size(3)
#         priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
#         bottle = self.bottleneck(torch.cat(priors, 1))
#         return self.relu(bottle)


class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.interpolate(self.conv(self.pool1(x)), size=(h, w), mode='bilinear', align_corners=True)
        self.layer2 = F.interpolate(self.conv(self.pool2(x)), size=(h, w), mode='bilinear', align_corners=True)
        self.layer3 = F.interpolate(self.conv(self.pool3(x)), size=(h, w), mode='bilinear', align_corners=True)
        self.layer4 = F.interpolate(self.conv(self.pool4(x)), size=(h, w), mode='bilinear', align_corners=True)

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out


#
class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class CE_Net_(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(CE_Net_, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = DACblock(512)
        self.spp = SPPblock(512)
        self.attention = attention(512)

        self.decoder4 = DecoderBlock(516, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e5 = self.dblock(e4)
        e5 = self.spp(e5)

        # Decoder
        d4 = self.decoder4(e5) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        final = torch.sigmoid(out)
        return final




class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x




class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.max_pool_conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 1024)
        self.up1 = up(1024+512+4, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)
        self.relu = nn.ReLU()

        self.dblock = DACblock(1024)
        self.spp = SPPblock(1024)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.dblock(x5)
        x5 = self.spp(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        # x = self.relu(x)
        return torch.sigmoid(x)

class double_conv1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv1, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, int(out_ch/2), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(out_ch/2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(out_ch/2), out_ch, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class pyramid_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(pyramid_conv, self).__init__()
        self.channel = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.channel1 = nn.Conv2d(out_ch, int(out_ch/2), kernel_size=1)
        self.channel2 = nn.Conv2d(out_ch, int(out_ch/4), kernel_size=1)
        self.channel3 = nn.Conv2d(out_ch, int(out_ch/4), kernel_size=1)
        self.down = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2)

        self.conv = double_conv1(int(out_ch/2),int(out_ch/2))
        self.convup = double_conv1(int(out_ch/4),int(out_ch/4))
        self.convdown = double_conv1(int(out_ch/4),int(out_ch/4))

    def forward(self, x):
        x = self.channel(x)
        x1 = self.channel1(x)
        x2 = self.channel2(x)
        x3 = self.channel3(x)
        x2 = self.up(x2)
        x3 = self.down(x3)

        xxx1 = self.conv(x1)
        xxx2 = self.convup(x2)
        xxx3 = self.convdown(x3)

        xx3 = self.up(xxx3)
        xx2 = self.down(xxx2)
        xx1 = xxx1
        xx = torch.cat([xx1,xx2,xx3],dim=1)
        #print(x.size(),xx.size())
        xxx = xx+x
        return xxx,xxx2,xxx1,xxx3
'''
        self.conv = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        )
        '''
class pyramid_conv_pyramid_supervison(nn.Module):
    def __init__(self, in_ch, out_ch,ch1,ch2,ch3):
        super(pyramid_conv_pyramid_supervison, self).__init__()
        self.channel = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.channel1 = nn.Conv2d(out_ch, int(out_ch/2), kernel_size=1)
        self.channel2 = nn.Conv2d(out_ch, int(out_ch/4), kernel_size=1)
        self.channel3 = nn.Conv2d(out_ch, int(out_ch/4), kernel_size=1)

        self.channel11 = nn.Sequential(nn.Conv2d(ch2,int(out_ch/2), kernel_size=1),nn.BatchNorm2d(int(out_ch/2)),nn.ReLU(inplace=True))
        self.channel22 = nn.Sequential(nn.Conv2d(ch1,int(out_ch/4), kernel_size=1),nn.BatchNorm2d(int(out_ch/4)),nn.ReLU(inplace=True))
        self.channel33 = nn.Sequential(nn.Conv2d(ch3,int(out_ch/4), kernel_size=1),nn.BatchNorm2d(int(out_ch/4)),nn.ReLU(inplace=True))

        self.channel111 = nn.Sequential(nn.Conv2d(int(out_ch/1),int(out_ch/2), kernel_size=1),nn.BatchNorm2d(int(out_ch/2)),nn.ReLU(inplace=True))
        self.channel222 = nn.Sequential(nn.Conv2d(int(out_ch/2),int(out_ch/4), kernel_size=1),nn.BatchNorm2d(int(out_ch/4)),nn.ReLU(inplace=True))
        self.channel333 = nn.Sequential(nn.Conv2d(int(out_ch/2),int(out_ch/4), kernel_size=1),nn.BatchNorm2d(int(out_ch/4)),nn.ReLU(inplace=True))
        
        self.down = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2)

        self.conv = double_conv1(int(out_ch/2),int(out_ch/2))
        self.convup = double_conv1(int(out_ch/4),int(out_ch/4))
        self.convdown = double_conv1(int(out_ch/4),int(out_ch/4))
        #print(out_ch)
        #print(ch1,ch2,ch3)
    def forward(self, x ,xxxx1,xxxx2,xxxx3):
        x = self.channel(x)
        x1 = self.channel1(x)
        x2 = self.channel2(x)
        x3 = self.channel3(x)
        x2 = self.up(x2)
        x3 = self.down(x3)
        '''
        #print('1',x1.size(),x2.size(),x3.size(),  xxxx2.size(),xxxx1.size(),xxxx3.size())
        x1 = torch.cat([x1,xxxx2],dim=1)
        x2 = torch.cat([x2,xxxx1],dim=1)
        x3 = torch.cat([x3,xxxx3],dim=1)
        
        #print(x1.size(),x2.size(),x3.size())

        x1 = self.channel11(x1)
        x2 = self.channel22(x2)
        x3 = self.channel33(x3)        
        '''
        xxxx1 = self.channel22(xxxx1)
        xxxx2 = self.channel11(xxxx2)
        xxxx3 = self.channel33(xxxx3)           

        x1 = torch.cat([x1,xxxx2],dim=1)
        x2 = torch.cat([x2,xxxx1],dim=1)
        x3 = torch.cat([x3,xxxx3],dim=1)      
          
        x1 = self.channel111(x1)
        x2 = self.channel222(x2)
        x3 = self.channel333(x3)      
        
        
                
        xx1 = self.conv(x1)
        xx2 = self.convup(x2)
        xx3 = self.convdown(x3)

        xx3 = self.up(xx3)
        xx2 = self.down(xx2)
        xx = torch.cat([xx1,xx2,xx3],dim=1)
        #print(x.size(),xx.size())
        xxx = xx+x
        return xxx


class downn(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(downn, self).__init__()
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            pyramid_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.max_pool_conv(x)
        return x


class upp(nn.Module):
    def __init__(self, in_ch, out_ch,ch1,ch2,ch3, bilinear=True):
        super(upp, self).__init__()

        self.up = nn.Upsample(scale_factor=2)
        self.conv = pyramid_conv_pyramid_supervison(in_ch, out_ch,ch1,ch2,ch3)

    def forward(self, x1,x3,x4,x5):
        x1 = self.up(x1)
        # x = torch.cat([x2, x1], dim=1)
        x= x1
        x = self.conv(x,x3,x4,x5)
        return x

class PUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(PUNet, self).__init__()
        self.inc = pyramid_conv(n_channels, 64)
        self.down1 = downn(64, 128)
        self.down2 = downn(128, 256)
        self.down3 = downn(256, 512)
        self.down4 = downn(512, 1024)
        self.up1 = upp(1024+4, 256,   288,576,640)
        self.up2 = upp(256, 128,           144,288,576)
        self.up3 = upp(128, 64,           64,144,288)
        self.up4 = upp(64, 64,           16,64,144)
        self.outc = outconv(64, n_classes)
        self.relu = nn.ReLU()

        self.dblock = DACblock(1024)
        self.spp = SPPblock(1024)


        self.upp2 = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
            nn.Upsample(scale_factor=2) )
        self.upp4 = nn.Sequential(
            nn.Conv2d(128, n_classes, kernel_size=1),
            nn.Upsample(scale_factor=4) )
        self.upp8 = nn.Sequential(
            nn.Conv2d(256, n_classes, kernel_size=1),
            nn.Upsample(scale_factor=8) )
        self.upp16 = nn.Sequential(
            nn.Conv2d(1024+4, n_classes, kernel_size=1),
            nn.Upsample(scale_factor=16) )

        self.supervision = nn.Sequential(
            nn.Conv2d(n_classes*5, n_classes*5, kernel_size=3, padding=1),
            nn.Conv2d(n_classes * 5, n_classes, kernel_size=1) )

    def forward(self, x):
        x1,x11,x12,x13  = self.inc(x)
        x2,x21,x22,x23 = self.down1(x1)
        x3,x31,x32,x33 = self.down2(x2)
        x4,x41,x42,x43 = self.down3(x3)
        x5,x51,x52,x53 = self.down4(x4)
        
        '''

        print('2',x21.size(),x22.size(),x23.size())
        print('3',x31.size(),x32.size(),x33.size())
        print('4',x41.size(),x42.size(),x43.size())
        print('5',x51.size(),x52.size(),x53.size())
        
        '''
        xxx1 = x11
        xxx2 = torch.cat([x12,x21],dim=1)
        xxx3 = torch.cat([x13,x22,x31],dim=1)
        xxx4 = torch.cat([x23,x32,x41],dim=1)
        xxx5 = torch.cat([x33,x42,x51],dim=1)
        xxx6 = torch.cat([x43,x52],dim=1)
        xxx7 = x53
        
        #print(xxx2.size(),xxx3.size(),xxx4.size(),xxx5.size(),xxx6.size(),xxx7.size())
        x5 = self.dblock(x5)
        xx6 = self.spp(x5)

        xx5 = self.up1(xx6, xxx4,xxx5,xxx6)
        xx4 = self.up2(xx5, xxx3,xxx4,xxx5)
        xx3 = self.up3(xx4, xxx2,xxx3,xxx4)
        xx2 = self.up4(xx3, xxx1,xxx2,xxx3)
        xx1 = self.outc(xx2)

        pyramid_pred2 = self.upp2(xx3)
        pyramid_pred4 = self.upp4(xx4)
        pyramid_pred8 = self.upp8(xx5)
        pyramid_pred16 = self.upp16(xx6)

        # print(xx1.size(),pyramid_pred2.size(),pyramid_pred4.size(),pyramid_pred8.size(),pyramid_pred16.size())
        xx = torch.cat([xx1,pyramid_pred2,pyramid_pred4,pyramid_pred8,pyramid_pred16],dim=1)
        xx = self.supervision(xx)

        # x = self.relu(x)
        # print(xx1.size(),xx.size())
        return torch.sigmoid(xx)
