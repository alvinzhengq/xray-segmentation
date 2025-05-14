import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module to capture multi-scale context."""
    def __init__(self, in_channels, out_channels=256, atrous_rates=(6, 12, 18)):
        super(ASPP, self).__init__()
        # 1x1 convolution branch (no dilation)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 3x3 atrous convolution branches with different dilation rates
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                      padding=atrous_rates[0], dilation=atrous_rates[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                      padding=atrous_rates[1], dilation=atrous_rates[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                      padding=atrous_rates[2], dilation=atrous_rates[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Global average pooling branch
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_pool = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Final 1x1 convolution to fuse all ASPP branch outputs
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Apply the parallel ASPP branches
        x1 = self.conv1(x)               # 1x1 conv branch
        x2 = self.conv2(x)               # 3x3 conv with dilation=6
        x3 = self.conv3(x)               # 3x3 conv with dilation=12
        x4 = self.conv4(x)               # 3x3 conv with dilation=18
        # Global feature branch: pool -> conv -> upsample to feature map size
        x5 = self.global_pool(x)         # (batch, channels, 1, 1)
        x5 = self.conv_pool(x5)          # (batch, channels, 1, 1)
        x5 = F.interpolate(x5, size=x4.shape[2:], mode='bilinear', align_corners=False)
        # Concatenate all branch outputs along the channel dimension
        x_cat = torch.cat([x1, x2, x3, x4, x5], dim=1)
        # Fuse the concatenated features with one more 1x1 conv
        return self.project(x_cat)