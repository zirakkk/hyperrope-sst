import torch
import torch.nn as nn
import torch.nn.functional as F

class SPCModuleIN(nn.Module):
    """Spectral Processing Module with Input Normalization - Original SSRN"""
    def __init__(self, in_channels, out_channels, bias=True):
        super(SPCModuleIN, self).__init__()
        self.s1 = nn.Conv3d(in_channels, out_channels, kernel_size=(7,1,1), stride=(2,1,1), bias=False)

    def forward(self, input):
        input = input.unsqueeze(1)  # Add channel dimension for 3D conv
        out = self.s1(input)
        return out.squeeze(1)  # Remove channel dimension

class SPAModuleIN(nn.Module):
    """Spatial Processing Module with Input Normalization - Original SSRN"""
    def __init__(self, in_channels, out_channels, k=49, bias=True):
        super(SPAModuleIN, self).__init__()
        self.s1 = nn.Conv3d(in_channels, out_channels, kernel_size=(k,3,3), padding=(0,1,1), bias=False)

    def forward(self, input):
        out = self.s1(input)
        out = torch.squeeze(out, 2)  # Remove spectral dimension
        return out

class ResSPC(nn.Module):
    """Residual Spectral Processing Block - Original SSRN"""
    def __init__(self, in_channels, out_channels, bias=True):
        super(ResSPC, self).__init__()
        
        self.spc1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(7,1,1), padding=(3,0,0), bias=False),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm3d(in_channels),
        )
        
        self.spc2 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(7,1,1), padding=(3,0,0), bias=False),
            nn.LeakyReLU(inplace=True),
        )
        
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, input):
        out = self.spc1(input)
        out = self.bn2(self.spc2(out))
        return F.leaky_relu(out + input)

class ResSPA(nn.Module):
    """Residual Spatial Processing Block - Original SSRN"""
    def __init__(self, in_channels, out_channels, bias=True):
        super(ResSPA, self).__init__()
        
        self.spa1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(in_channels),
        )
        
        self.spa2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        out = self.spa1(input)
        out = self.bn2(self.spa2(out))
        return F.leaky_relu(out + input)

class SSRN(nn.Module):
    """
    Paper-faithful SSRN implementation based on:
    "Spectral-Spatial Residual Network for Hyperspectral Image Classification: A 3-D Deep Learning Framework"
    by Zilong Zhong et al., IEEE T-GRS 2018
    """
    def __init__(self, params):
        super(SSRN, self).__init__()
        
        # Extract parameters
        num_classes = params['data'].get('num_classes', 9)
        k = params['data'].get('k', 49)
        
        
        channels = 8
        
        # Architecture layers (exactly as in original paper)
        self.layer1 = SPCModuleIN(1, channels)
        self.layer2 = ResSPC(channels, channels)
        self.layer3 = ResSPC(channels, channels)
        self.layer4 = SPAModuleIN(channels, channels, k=k)
        self.bn4 = nn.BatchNorm2d(channels)
        self.layer5 = ResSPA(channels, channels)
        self.layer6 = ResSPA(channels, channels)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x, _=None, __=None):
        """
        Forward pass - exactly as in original SSRN paper
        Input: x with shape (batch_size, spectral_bands, height, width)
        """
        # Spectral processing (3D CNN)
        x = F.leaky_relu(self.layer1(x))
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Spatial processing (2D CNN)
        x = self.bn4(F.leaky_relu(self.layer4(x)))
        x = self.layer5(x)
        x = self.layer6(x)
        
        # Classification
        x = F.avg_pool2d(x, x.size()[-1])
        x = self.fc(x.squeeze())
        
        return x
