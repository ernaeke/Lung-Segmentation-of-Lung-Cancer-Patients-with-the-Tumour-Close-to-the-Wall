import torch
import torch.nn as nn

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

class UNet3D(nn.Module):
    def __init__(self):
        super(UNet3D, self).__init__()

        # Encoder
        self.encoder1 = conv_block(1, 32)
        self.pool1 = nn.MaxPool3d(2)

        self.encoder2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool3d(2)

        self.encoder3 = conv_block(64, 128)
        self.pool3 = nn.MaxPool3d(2)

        self.encoder4 = conv_block(128, 256)
        self.pool4 = nn.MaxPool3d(2)

        self.bottleneck = conv_block(256, 512)

        # Decoder
        self.up4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = conv_block(512, 256)

        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = conv_block(256, 128)

        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = conv_block(128, 64)

        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = conv_block(64, 32)

        self.final = nn.Conv3d(32, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        e4 = self.encoder4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b)
        e4 = self._crop_to_match(e4, d4)
        d4 = self.decoder4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        e3 = self._crop_to_match(e3, d3)
        d3 = self.decoder3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        e2 = self._crop_to_match(e2, d2)
        d2 = self.decoder2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        e1 = self._crop_to_match(e1, d1)
        d1 = self.decoder1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.final(d1))

    def _crop_to_match(self, enc_feat, target_feat):
        """Crop encoder feature map to match upsampled decoder feature map in spatial size."""
        diffZ = enc_feat.shape[2] - target_feat.shape[2]
        diffY = enc_feat.shape[3] - target_feat.shape[3]
        diffX = enc_feat.shape[4] - target_feat.shape[4]

        return enc_feat[
            :,
            :,
            diffZ // 2 : diffZ // 2 + target_feat.shape[2],
            diffY // 2 : diffY // 2 + target_feat.shape[3],
            diffX // 2 : diffX // 2 + target_feat.shape[4]
        ]
