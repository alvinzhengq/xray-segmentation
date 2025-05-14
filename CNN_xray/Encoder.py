import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNet50Encoder(nn.Module):
    def __init__(self, pretrained=True):
        """
        ResNet-50 encoder for 1-channel (grayscale) input.
        Returns feature maps at multiple scales for decoder skip connections.
        """
        super(ResNet50Encoder, self).__init__()

        resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        
        # Original ResNet-50 conv1: in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            # Initialize conv1 weights from the pretrained model by averaging the RGB weights.
            # resnet50.conv1.weight shape: [64, 3, 7, 7] -> take mean across the 3 input channels to get [64, 1, 7, 7]
            with torch.no_grad():
                self.conv1.weight.copy_(resnet50.conv1.weight.mean(dim=1, keepdim=True))
        
        # Keep the remaining layers of ResNet-50 unchanged
        self.bn1 = resnet50.bn1        
        self.relu = resnet50.relu      
        self.maxpool = resnet50.maxpool 
        

        self.layer1 = resnet50.layer1  # outputs 256 channels, stride 1 (same spatial size as after maxpool)
        self.layer2 = resnet50.layer2  # outputs 512 channels, stride 2 (spatial size halved)
        self.layer3 = resnet50.layer3  # utputs 1024 channels, stride 2
        self.layer4 = resnet50.layer4  # outputs 2048 channels, stride 2

    
    def forward(self, x):
        """
        Forward pass for the encoder.
        x: input tensor of shape [batch_size, 1, 256, 256] (grayscale images).
        Returns a tuple of feature maps (c1, c2, c3, c4, c5) from different stages.
        """
        # Initial convolution + BatchNorm + ReLU
        x = self.conv1(x)   # shape: [batch, 64, 128, 128] after conv1 (7x7 stride 2 on 256x256 input)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = x  # Feature map after first conv block (low-level features, 64 channels, 128x128 resolution)
        
        # Max pooling reduces spatial size by 2
        x = self.maxpool(x)  # shape: [batch, 64, 64, 64] after 3x3 max pool
        
        # Pass through ResNet layers
        c2 = self.layer1(x)  # Output of layer1 (256 channels, 64x64 resolution)
        c3 = self.layer2(c2) # Output of layer2 (512 channels, 32x32 resolution)
        c4 = self.layer3(c3) # Output of layer3 (1024 channels, 16x16 resolution)
        c5 = self.layer4(c4) # Output of layer4 (2048 channels, 8x8 resolution)
        
        # Return multi-scale feature maps for decoder
        return c1, c2, c3, c4, c5
