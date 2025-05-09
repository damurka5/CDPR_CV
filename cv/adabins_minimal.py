# adabins_minimal.py
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

class UnetAdaptiveBins(nn.Module):
    def __init__(self, n_bins=256, min_val=0.1, max_val=10.0, norm='linear'):
        super(UnetAdaptiveBins, self).__init__()
        self.num_classes = n_bins
        self.min_val = min_val
        self.max_val = max_val
        self.encoder = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                    in_channels=3, out_channels=128, init_features=32)
        
        self.conv_out = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, n_bins, kernel_size=1, stride=1, padding=0)
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        x = self.encoder(x)
        return self.conv_out(x)