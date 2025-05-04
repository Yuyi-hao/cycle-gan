import torch.nn as nn

class PatchDiscriminator(nn.Module):
    def __init__(self, input_nc):
        super(PatchDiscriminator, self).__init__()
        
        def conv_layer(in_c, out_c, stride):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1),
                nn.InstanceNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.model = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            conv_layer(64, 128, stride=2),
            conv_layer(128, 256, stride=2),
            conv_layer(256, 512, stride=1),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)
