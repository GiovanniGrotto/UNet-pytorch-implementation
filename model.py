import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, channels=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down passage
        for channel in channels:
            self.downs.append(DoubleConv(in_channels, channel))
            in_channels = channel

        # Up passage
        for channel in reversed(channels):
            self.ups.append(
                nn.ConvTranspose2d(channel*2, channel, kernel_size=2, stride=2)
            )
            self.ups.append((DoubleConv(channel*2, channel)))

        self.bottleneck = DoubleConv(channels[-1], channels[-1]*2)
        self.last_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Dobbiamo fare questa porcata per via della concatenzione tra discesa e salita
        for idx in range(0, len(self.ups), 2):
            up = self.ups[idx]
            x = up(x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            double_conv = self.ups[idx+1]
            x = double_conv(concat_skip)

        return self.last_conv(x)


def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNet(in_channels=1, out_channels=1)
    preds = model(x)
    print(x.shape)
    print(preds.shape)
    assert x.shape == preds.shape


if __name__ == '__main__':
    test()
