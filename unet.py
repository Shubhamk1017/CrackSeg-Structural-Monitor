"""
U-Net Architecture for Crack Segmentation
==========================================
Built from scratch following the original U-Net paper:
"U-Net: Convolutional Networks for Biomedical Image Segmentation"
by Ronneberger et al., 2015 (https://arxiv.org/abs/1505.04597)

The key idea of U-Net is the symmetric encoder-decoder structure with
skip connections. The encoder gradually reduces spatial dimensions while
learning features, and the decoder upsamples back to the original resolution.
Skip connections directly connect encoder layers to decoder layers so that
the decoder can recover fine spatial details that get lost during downsampling.

I chose to build this from scratch instead of using a library because
I wanted to understand every part of the architecture.

Architecture:
    Input (3, 256, 256)
        |
    [Encoder]  64 -> 128 -> 256 -> 512
        |
    [Bottleneck] 1024
        |
    [Decoder]  512 -> 256 -> 128 -> 64
        |
    Output (1, 256, 256)

- Shubham Kumar, Summer of Science 2026
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    The basic building block of U-Net: two 3x3 convolutions, each followed
    by batch normalization and ReLU activation.

    Why batch normalization? It normalizes the activations between layers,
    which stabilizes training and lets us use higher learning rates.
    Without it, training deep networks like U-Net is much harder.

    Why two convolutions? The original paper uses two 3x3 convs per block.
    Two 3x3 convs have the same receptive field as one 5x5 conv but with
    fewer parameters and more non-linearity (two ReLUs instead of one).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Note: bias=False because BatchNorm already has a learnable bias term,
        # so the conv bias would be redundant

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Encoder block: MaxPool to halve spatial dimensions, then DoubleConv.
    
    MaxPool2d(2) reduces the feature map from (H, W) to (H/2, W/2).
    This lets the network "see" a larger area of the input image in
    deeper layers (increasing the receptive field).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Decoder block: upsample, concatenate with skip connection, then DoubleConv.
    
    I'm using ConvTranspose2d for upsampling because it's a learned upsampling -
    the network can learn the best way to increase resolution, instead of just
    using bilinear interpolation which is fixed.
    
    The skip connection is the key part: we concatenate the features from the
    corresponding encoder layer. This gives the decoder access to both:
    1. High-level semantic features from the deeper layers (what is it?)
    2. Fine spatial details from the encoder (where exactly is it?)
    
    This is especially important for crack segmentation because cracks are
    thin structures, and we need precise spatial localization.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # ConvTranspose2d doubles the spatial dimensions
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                      kernel_size=2, stride=2)
        # After concatenation with skip connection, we have
        # in_channels//2 (from upsample) + in_channels//2 (from skip) = in_channels
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        # Upsample the decoder features
        x = self.up(x)

        # Concatenate with the skip connection from the encoder
        # dim=1 means concatenate along the channel dimension
        x = torch.cat([skip, x], dim=1)

        return self.conv(x)


class UNet(nn.Module):
    """
    Complete U-Net model for binary segmentation.
    
    For our crack segmentation task:
    - Input: RGB image (3 channels, 256x256)
    - Output: single channel prediction map (1 channel, 256x256)
      where each pixel value represents the probability of being a crack
    
    The output is raw logits (no sigmoid), because we use BCEWithLogitsLoss
    during training which applies sigmoid internally and is more numerically
    stable than doing sigmoid + BCELoss separately.
    """
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # ===== ENCODER (contracting path) =====
        # Each level doubles the number of feature channels
        # and halves the spatial dimensions

        self.inc = DoubleConv(in_channels, 64)     # 256x256 -> 256x256, 64 channels
        self.down1 = Down(64, 128)                  # 256x256 -> 128x128, 128 channels
        self.down2 = Down(128, 256)                 # 128x128 -> 64x64, 256 channels
        self.down3 = Down(256, 512)                 # 64x64 -> 32x32, 512 channels

        # ===== BOTTLENECK =====
        self.down4 = Down(512, 1024)                # 32x32 -> 16x16, 1024 channels

        # ===== DECODER (expanding path) =====
        # Each level halves the channels and doubles spatial dimensions
        # Skip connections bring back fine details from the encoder

        self.up1 = Up(1024, 512)                    # 16x16 -> 32x32, 512 channels
        self.up2 = Up(512, 256)                     # 32x32 -> 64x64, 256 channels
        self.up3 = Up(256, 128)                     # 64x64 -> 128x128, 128 channels
        self.up4 = Up(128, 64)                      # 128x128 -> 256x256, 64 channels

        # ===== OUTPUT =====
        # 1x1 convolution maps 64 feature channels to our output channels
        # It's basically a learned per-pixel linear combination of features
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder - save outputs for skip connections
        x1 = self.inc(x)      # level 1 features (256x256, 64ch)
        x2 = self.down1(x1)   # level 2 features (128x128, 128ch)
        x3 = self.down2(x2)   # level 3 features (64x64, 256ch)
        x4 = self.down3(x3)   # level 4 features (32x32, 512ch)

        # Bottleneck
        x5 = self.down4(x4)   # bottleneck (16x16, 1024ch)

        # Decoder - use skip connections from encoder
        x = self.up1(x5, x4)  # combine bottleneck with level 4
        x = self.up2(x, x3)   # combine with level 3
        x = self.up3(x, x2)   # combine with level 2
        x = self.up4(x, x1)   # combine with level 1

        # Final 1x1 conv to get output
        out = self.outc(x)
        return out


# Quick test to verify the architecture works
if __name__ == '__main__':
    # Create a dummy input: batch of 2 RGB images, 256x256
    dummy_input = torch.randn(2, 3, 256, 256)
    
    model = UNet(in_channels=3, out_channels=1)
    output = model(dummy_input)
    
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
