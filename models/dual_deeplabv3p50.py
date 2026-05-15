# 最终版网络结构，不要再跑错了！
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mobilenetv2 import mobilenetv2
from models.ECA import ECAAttention
from models.CoordAtt import CoordAtt 
from segmentation_models_pytorch.encoders import get_encoder

class MobileNetV2(nn.Module):
    def __init__(self, in_channels=3, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial
        
        model = mobilenetv2(pretrained)

        # 修改通道参数为输入通道数
        model.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        self.features   = model.features[:-1]

        self.total_idx  = len(self.features)
        self.down_idx   = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
        
    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x 


class SMPResNetBackbone(nn.Module):
    """
    用 segmentation_models_pytorch 的 ResNet encoder 封装成:
        low_level_features, x
    的形式，接口对齐 MobileNetV2。
    """
    def __init__(self, encoder_name="resnet34",
                 in_channels=3,
                 downsample_factor=16,
                 pretrained=True):
        super(SMPResNetBackbone, self).__init__()

        # smp 里 pretrained 对应的是 weights 字符串
        if pretrained:
            weights = "imagenet"
        else:
            weights = None

        # depth=5 => 返回 5 个尺度的 features: [H/2, H/4, H/8, H/16, H/32]
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=weights
        )

        self.out_channels       = self.encoder.out_channels[-1]  # 最深层通道
        self.low_level_channels = self.encoder.out_channels[1]   # 低层特征（通常 H/4）

        # 控制输出 stride（相当于你的 downsample_factor）
        # 原始 ResNet 输出 stride=32，这里用 dilation 改成 16 或 8
        assert downsample_factor in [8, 16], "只支持 downsample_factor=8 或 16"

        if downsample_factor == 16:
            # 最后一层膨胀，输出 stride=16
            # stage_list=[5] 对应最后一层
            self.encoder.make_dilated(
                stage_list=[5],
                dilation_list=[2],
            )
        elif downsample_factor == 8:
            # 后两层都膨胀，输出 stride=8
            self.encoder.make_dilated(
                stage_list=[4, 5],
                dilation_list=[2, 4],
            )

    def forward(self, x):
        """
        smp encoder 的 forward 返回一个 list[features]:
            features[0]: H/2
            features[1]: H/4  -> 适合作为 low_level
            ...
            features[-1]: H/8 或 H/16（取决于上面的 make_dilated）
        """
        feats = self.encoder(x)
        low_level = feats[1]   # H/4
        x = feats[-1]          # 主干最后的高层特征
        return low_level, x


#-----------------------------------------#
#   Residual模块
#   残差密集块，用于对AMSR2进行特征提取
#-----------------------------------------#
class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out += residual
        out = self.relu(out)
        return out


#-----------------------------------------#
#   添加了ECA的Residual模块
#   残差密集块，用于对AMSR2进行特征提取
#-----------------------------------------#
class ECAResidualBlock(nn.Module):      # 左侧的 residual block 结构（18-layer、34-layer）
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):      # 两层卷积 Conv2d + Shutcuts
        super(ECAResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # self.attention = EfficientChannelAttention(planes)       # Efficient Channel Attention module
        self.attention = ECAAttention(kernel_size=3)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:      # Shutcuts用于构建 Conv Block 和 Identity Block
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        ECA_out = self.attention(out)  # 注意力机制加权后的特征图
        out = ECA_out + self.shortcut(x)  # 残差连接
        out = F.relu(out)  # 再次应用ReLU激活函数
        return out
    

#-----------------------------------------#
#   ASPP特征提取模块
#   利用不同膨胀率的膨胀卷积进行特征提取
#-----------------------------------------#
class ASPP(nn.Module):
    def __init__(self, dim_in_hr, dim_in_lr, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
                nn.Conv2d(dim_in_hr, dim_out, 1, 1, padding=0, dilation=rate,bias=True),
                nn.BatchNorm2d(dim_out, momentum=bn_mom),
                nn.ReLU(inplace=True),
        )  # 1×1
        self.branch2 = nn.Sequential(
                nn.Conv2d(dim_in_hr, dim_out, 3, 1, padding=6*rate, dilation=6*rate, bias=True),
                nn.BatchNorm2d(dim_out, momentum=bn_mom),
                nn.ReLU(inplace=True),
        )  # 3×3 扩张率6
        self.branch3 = nn.Sequential(
                nn.Conv2d(dim_in_hr, dim_out, 3, 1, padding=12*rate, dilation=12*rate, bias=True),
                nn.BatchNorm2d(dim_out, momentum=bn_mom),
                nn.ReLU(inplace=True),
        )  # 3×3 扩张率12
        self.branch4 = nn.Sequential(
                nn.Conv2d(dim_in_hr, dim_out, 3, 1, padding=18*rate, dilation=18*rate, bias=True),
                nn.BatchNorm2d(dim_out, momentum=bn_mom),
                nn.ReLU(inplace=True),
        )  # 3×3 扩张率18
        # Image Pooling（这里分为三步）
        self.branch5_conv = nn.Conv2d(dim_in_hr, dim_out, 1, 1, 0,bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
                nn.Conv2d(dim_out*5+dim_in_lr, dim_out, 1, 1, padding=0,bias=True),
                nn.BatchNorm2d(dim_out, momentum=bn_mom),
                nn.ReLU(inplace=True),
        )

        self.res_block =  ResidualConvBlock(in_channels=dim_in_lr, out_channels=dim_in_lr, kernel_size=3, stride=1, padding=1)
        self.eca_res_block = ECAResidualBlock(in_planes=dim_in_lr, planes=dim_in_lr, stride=1)

    def forward(self, x, y):
        [b, c, xrow, xcol] = x.size()
        #-----------------------------------------#
        #   一共五个分支
        #-----------------------------------------#
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        #-----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        #-----------------------------------------#
        global_feature = torch.mean(x,2,True)
        global_feature = torch.mean(global_feature,3,True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (xrow, xcol), None, 'bilinear', True)
        # -----------------------------------------#
        #   将AMSR2数据通过ResidualConvBlock处理后再上采样
        # -----------------------------------------#
        # 假设y的通道数与ResidualConvBlock的输入通道数匹配
        # 设置多个残差块
        am_feature = self.res_block(y)
        am_feature = self.eca_res_block(am_feature)
        # 上采样
        am_feature = F.interpolate(am_feature, (xrow, xcol), None, 'bilinear', True)
        #-----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        #-----------------------------------------#
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature, am_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result


class DeepLab(nn.Module):
    def __init__(self, options):
        self.opt = options
        self.nclass = options.get('nclass', 11)
        self.use_iw_head = options.get('use_iw_head', True)

        # ---- 输入通道与切片 ---- #
        self.sar_in   = options.get('sar_in_channels', 3)     # 前3
        self.amsr_in  = options.get('amsr_in_channels', 6)    # 后6

        self.backbone = options['backbone']
        self.pretrained = options['pretrained']
        self.downsample_factor = options['downsample_factor']
        
        super(DeepLab, self).__init__()
        if self.backbone=="mobilenet":
            #----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,24]
            #   主干部分    [30,30,320]
            #----------------------------------#
            self.backbone = MobileNetV2(in_channels=self.sar_in, downsample_factor=self.downsample_factor, pretrained=self.pretrained)
            in_channels = 320
            low_level_channels = 24
        elif self.backbone_name in ["resnet18", "resnet34", "resnet50", "resnet101"]:
            #----------------------------------#
            #   ResNet 主干（smp）
            #----------------------------------#
            self.backbone = SMPResNetBackbone(
                encoder_name=self.backbone_name,
                in_channels=self.sar_in,
                downsample_factor=self.downsample_factor,
                pretrained=self.pretrained
            )
            in_channels       = self.backbone.out_channels
            low_level_channels = self.backbone.low_level_channels
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(self.backbone))

        #-----------------------------------------#
        #   ASPP特征提取模块
        #   利用不同膨胀率的膨胀卷积进行特征提取
        #-----------------------------------------#
        self.aspp = ASPP(dim_in_hr=in_channels, dim_in_lr=self.amsr_in, dim_out=256, rate=16//self.downsample_factor)
        
        #----------------------------------#
        #   浅层特征边
        #----------------------------------#
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )        

        self.cat_conv = nn.Sequential(
            nn.Conv2d(48+256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )

        self.sic_head = nn.Conv2d(256, self.nclass, 1, stride=1)
        if self.use_iw_head:
            self.iw_head = nn.Conv2d(256, 2, 1, stride=1)

        self.coordatt1 = CoordAtt(48, 48)
        self.coordatt2 = CoordAtt(256, 256)

    def forward(self, x, y, mode='all', **kwargs):
        H, W = x.size(2), x.size(3)
        #-----------------------------------------#
        #   获得两个特征层
        #   low_level_features: 浅层特征-进行卷积处理
        #   x : 主干部分-利用ASPP结构进行加强特征提取
        #-----------------------------------------#
        low_level_features, x = self.backbone(x)
        x = self.aspp(x, y)
        low_level_features = self.shortcut_conv(low_level_features)
        low_level_features = self.coordatt1(low_level_features)
        
        #-----------------------------------------#
        #   将加强特征边上采样
        #   与浅层特征堆叠后利用卷积进行特征提取
        #-----------------------------------------#
        x = self.coordatt2(x)
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        
        out_dict = {}

        if mode in ['sic_only', 'all']:
            y_sic = self.sic_head(x)
            y_sic = F.interpolate(y_sic, size=(H, W), mode='bilinear', align_corners=True)
            out_dict['label_80_sic'] = [y_sic]

        if self.use_iw_head and mode in ['iw_only', 'all']:
            y_iw = self.iw_head(x)
            y_iw = F.interpolate(y_iw, size=(H, W), mode='bilinear', align_corners=True)
            out_dict['label_manual'] = [y_iw]

        return out_dict
