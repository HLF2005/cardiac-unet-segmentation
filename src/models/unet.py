
import torch
import torch.nn as nn

def convolution_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def upsampling_block(in_channels, out_channels):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=2,
        stride=2,
    )

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Encoder
        self.encoder1 = convolution_block(in_channels, 64)
        self.encoder2 = convolution_block(64, 128)
        self.encoder3 = convolution_block(128, 256)
        self.encoder4 = convolution_block(256, 512)

        self.max_pool = nn.MaxPool2d(kernel_size=2)

        self.bottleneck = convolution_block(512, 1024)

        # Decoder
        self.upsample4 = upsampling_block(1024, 512)
        self.decoder4 = convolution_block(1024, 512)

        self.upsample3 = upsampling_block(512, 256)
        self.decoder3 = convolution_block(512, 256)

        self.upsample2 = upsampling_block(256, 128)
        self.decoder2 = convolution_block(256, 128)

        self.upsample1 = upsampling_block(128, 64)
        self.decoder1 = convolution_block(128, 64)

        self.output_layer = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):

        encoder1_features = self.encoder1(x)
        encoder2_features = self.encoder2(self.max_pool(encoder1_features))
        encoder3_features = self.encoder3(self.max_pool(encoder2_features))
        encoder4_features = self.encoder4(self.max_pool(encoder3_features))

        bottleneck_features = self.bottleneck(self.max_pool(encoder4_features))

        decoder4_features = self.upsample4(bottleneck_features)
        decoder4_features = self.decoder4(
            torch.cat([decoder4_features, encoder4_features], dim=1)
        )

        decoder3_features = self.upsample3(decoder4_features)
        decoder3_features = self.decoder3(
            torch.cat([decoder3_features, encoder3_features], dim=1)
        )

        decoder2_features = self.upsample2(decoder3_features)
        decoder2_features = self.decoder2(
            torch.cat([decoder2_features, encoder2_features], dim=1)
        )

        decoder1_features = self.upsample1(decoder2_features)
        decoder1_features = self.decoder1(
            torch.cat([decoder1_features, encoder1_features], dim=1)
        )

        return self.output_layer(decoder1_features)
