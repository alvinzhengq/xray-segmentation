import torch.nn as nn
from Encoder import ResNet50Encoder
from ASPP import ASPP
from Decoder import Decoder

class MultiLabelSegModel(nn.Module):
    def __init__(self):
        super(MultiLabelSegModel, self).__init__()
        # Initialize the encoder, ASPP, and decoder
        self.encoder = ResNet50Encoder(pretrained=True) 
        self.aspp = ASPP(in_channels=2048, out_channels=256) 
        self.decoder = Decoder(num_classes=118)
    
    def forward(self, x):
        # x shape: (N, 1, 256, 256)

        c1, c2, c3, c4, c5 = self.encoder(x)
        logits = self.decoder(c1, c2, c3, c4, c5)
        return logits  # raw logits (shape: N x 118 x 256 x 256)
