# Key modules definition

import torch.nn as nn

class Conv_BN_NLA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super(Conv_BN_NLA, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        activation_maps = self.conv(x)
        normalized_activation_maps = self.bn(activation_maps)
        return self.activation(normalized_activation_maps)


class Linear_BN_NLA(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear_BN_NLA, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(num_features=out_features)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        z1 = self.linear(x)
        z2 = self.bn(z1)
        return self.activation(z2)


class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()  # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


class CircleNet(nn.Module):
    def __init__(self):
        super(CircleNet, self).__init__()
        self.architecture = nn.Sequential(
            nn.BatchNorm2d(num_features=3),
            Conv_BN_NLA(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            Conv_BN_NLA(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            Conv_BN_NLA(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            Conv_BN_NLA(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            Conv_BN_NLA(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            Conv_BN_NLA(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            Conv_BN_NLA(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            Conv_BN_NLA(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            Conv_BN_NLA(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            Conv_BN_NLA(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            Conv_BN_NLA(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            Conv_BN_NLA(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Dropout2d(),

            Flatten(),
            Linear_BN_NLA(in_features=2 * 2 * 512, out_features=1024),
            Linear_BN_NLA(in_features=1024, out_features=1024),
            nn.Linear(in_features=1024, out_features=3),
        )

    def forward(self, x):
        return self.architecture(x)
