import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    A basic convolutional block: Conv2D -> (BatchNorm) -> (Activation).

    - If `discriminator=True`, uses LeakyReLU.
    - Otherwise, uses PReLU.
    - BatchNorm can be toggled with `use_bn`.
    - Activation can be toggled with `use_act`.

    Input:  (B, in_channels, H, W)
    Output: (B, out_channels, H, W)
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        discriminator=False,
        use_act=True,
        use_bn=True,
        **kwargs
    ):
        super().__init__()
        self.use_act = use_act
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not use_bn, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = (
            nn.LeakyReLU(0.2, inplace=True) if discriminator
            else nn.PReLU(num_parameters=out_channels)
        )

    def forward(self, X):
        if self.use_act:
            return self.act(self.bn(self.conv(X)))
        else:
            return self.bn(self.conv(X))


class UpsampleBlock(nn.Module):
    """
    Upsampling block using PixelShuffle.

    Conv2D increases channels -> PixelShuffle rearranges -> PReLU activation.

    Args:
        in_channels (int): Number of input channels.
        scale_factor (int): Upscaling factor (e.g., 2 or 4).

    Input:  (B, in_channels, H, W)
    Output: (B, in_channels, H*scale, W*scale)
    """
    def __init__(self, in_channels, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels * scale_factor ** 2,
            kernel_size=3, stride=1, padding=1
        )
        self.ps = nn.PixelShuffle(scale_factor)
        self.act = nn.PReLU(num_parameters=in_channels)

    def forward(self, X):
        return self.act(self.ps(self.conv(X)))


class ResidualBlock(nn.Module):
    """
    Residual block with two ConvBlocks.

    Structure: ConvBlock -> ConvBlock -> Residual Add.

    Input:  (B, C, H, W)
    Output: (B, C, H, W)
    """
    def __init__(self, in_channels):
        super().__init__()
        self.block1 = ConvBlock(
            in_channels, in_channels,
            kernel_size=3, stride=1, padding=1
        )
        self.block2 = ConvBlock(
            in_channels, in_channels,
            kernel_size=3, stride=1, padding=1,
            use_act=False
        )

    def forward(self, X):
        out = self.block1(X)
        out = self.block2(out)
        return out + X


class Generator(nn.Module):
    """
    Generator network for SRGAN.

    - Initial ConvBlock (no BN)
    - Stack of ResidualBlocks
    - ConvBlock + skip connection
    - Upsampling (PixelShuffle blocks)
    - Final Conv2D + Tanh

    Input:  (B, in_channels, H, W)
    Output: (B, out_channels, H*4, W*4)   # since two upsample x2
    """
    def __init__(self, in_channels=3, out_channels=3, num_channels=64, num_blocks=16):
        super().__init__()
        self.initial = ConvBlock(in_channels, num_channels, kernel_size=9, stride=1, padding=4, use_bn=False)
        self.residuals = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.convblock = ConvBlock(num_channels, num_channels, kernel_size=3, stride=1, padding=1, use_act=False)
        self.upsamples = nn.Sequential(
            UpsampleBlock(num_channels, 2),
            UpsampleBlock(num_channels, 2)
        )
        self.final = nn.Conv2d(num_channels, out_channels, kernel_size=9, stride=1, padding=4)

    def forward(self, X):
        initial = self.initial(X)
        X = self.residuals(initial)
        X = self.convblock(X) + initial  # skip connection
        X = self.upsamples(X)
        return torch.tanh(self.final(X))


class Discriminator(nn.Module):
    """
    Discriminator network for SRGAN.

    - Series of ConvBlocks with increasing channels.
    - Ends with adaptive pooling and fully connected layers.

    Input:  (B, in_channels, H, W)
    Output: (B, 1)  # real/fake score
    """
    def __init__(self, in_channels=3, features=[64, 64, 128, 128, 256, 256, 512, 512]):
        super().__init__()
        blocks = []
        for idx, feature in enumerate(features):
            blocks.append(
                ConvBlock(
                    in_channels, feature,
                    kernel_size=3,
                    stride=1 + idx % 2,  # alternate strides for downsampling
                    padding=1,
                    discriminator=True,
                    use_act=True,
                    use_bn=False if idx == 0 else True
                )
            )
            in_channels = feature  # update for next block

        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512*6*6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )

    def forward(self, X):
        X = self.blocks(X)
        return self.classifier(X)