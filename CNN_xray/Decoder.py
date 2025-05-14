import torch
import torch.nn as nn
import torch.nn.functional as F
from ASPP import ASPP

class Decoder(nn.Module):
    """
    U-Net style decoder with skip connections and ASPP.
    Takes encoder feature maps c1 (highest resolution) through c5 (lowest resolution),
    and up-samples to produce a segmentation mask.
    """
    def __init__(self, c1_channels=64, c2_channels=256, c3_channels=512, 
                 c4_channels=1024, c5_channels=2048, num_classes=118):
        super(Decoder, self).__init__()
        # ASPP module at the bottleneck (after c5) to capture multi-scale context
        self.aspp = ASPP(in_channels=c5_channels, out_channels=256)
        # Decoder blocks: Conv layers to fuse upsampled features with skip connections
        self.up4_conv = nn.Sequential(  # combines ASPP output (256) + c4 (skip)
            nn.Conv2d(256 + c4_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.up3_conv = nn.Sequential(  # combines up4 output + c3
            nn.Conv2d(256 + c3_channels, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up2_conv = nn.Sequential(  # combines up3 output + c2
            nn.Conv2d(128 + c2_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.up1_conv = nn.Sequential(  # combines up2 output + c1
            nn.Conv2d(64 + c1_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Final 1x1 convolution to get the desired number of classes
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, c1, c2, c3, c4, c5):
        """
        c5: bottleneck feature map (lowest resolution, deepest layer of encoder)
        c4, c3, c2, c1: skip feature maps from encoder (progressively higher resolution)
        """
        # ASPP at bottleneck
        x = self.aspp(c5)  
        # Decoder Stage 4: Upsample bottleneck output and merge with c4 skip features
        x = F.interpolate(x, size=c4.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, c4], dim=1)          # concatenate skip (c4) with upsampled features
        x = self.up4_conv(x)                  # conv fusion (output has 256 channels)
        # Decoder Stage 3: Upsample and merge with c3 skip
        x = F.interpolate(x, size=c3.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, c3], dim=1)
        x = self.up3_conv(x)                  # output has 128 channels
        # Decoder Stage 2: Upsample and merge with c2 skip
        x = F.interpolate(x, size=c2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, c2], dim=1)
        x = self.up2_conv(x)                  # output has 64 channels
        # Decoder Stage 1: Upsample and merge with c1 skip (highest resolution encoder feature)
        x = F.interpolate(x, size=c1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, c1], dim=1)
        x = self.up1_conv(x)                  # output has 64 channels
        # Final upsampling to original image size (256x256) 
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        # Output segmentation map with 118 channels, apply sigmoid for multi-label probabilities
        x = self.final_conv(x)
        #return torch.sigmoid(x)
        return x