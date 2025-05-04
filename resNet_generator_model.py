import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
        )
    
    def forward(self, x):
        return x+self.block(x)

class ResNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, n_blocks=9):
        super(ResNetGenerator, self).__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        # downsampling
        in_feature = 64
        out_feature = in_feature*2
        for _ in range(2):
            model += [
                nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_feature),
                nn.ReLU(inplace=True)
            ]
            in_feature = out_feature
            out_feature *= 2
        
        # residual block
        for _ in range(n_blocks):
            model += [ResidualBlock(in_feature)]

        # upsampling 
        out_feature = in_feature // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_feature, out_feature, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_feature),
                nn.ReLU(inplace=True)
            ]
            in_feature = out_feature
            out_feature = in_feature // 2
        
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    
    def forward(self, x):
        return x+self.model(x)

    