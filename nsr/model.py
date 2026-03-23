import torch
import torch.nn as nn
import MinkowskiEngine as ME


class ResBlock(ME.MinkowskiNetwork):
    def __init__(self, in_channels, out_channels, dimension=3):
        super(ResBlock, self).__init__(dimension)
        self.conv1 = ME.MinkowskiConvolution(
            in_channels, out_channels, kernel_size=3, stride=1, dimension=dimension
        )
        self.norm1 = ME.MinkowskiBatchNorm(out_channels)
        self.relu = ME.MinkowskiReLU()

        self.conv2 = ME.MinkowskiConvolution(
            out_channels, out_channels, kernel_size=3, stride=1, dimension=dimension
        )
        self.norm2 = ME.MinkowskiBatchNorm(out_channels)

        if in_channels != out_channels:
            self.downsample = ME.MinkowskiConvolution(
                in_channels, out_channels, kernel_size=1, stride=1, dimension=dimension
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class RecurrentUNet(ME.MinkowskiNetwork):
    def __init__(self, in_channels=7, out_channels=1, D=4, prune_threshold=0.0):
        super(RecurrentUNet, self).__init__(D)
        self.prune_threshold = float(prune_threshold)
        spatial_stride = (2, 2, 2, 1)

        # --- Encoder ---
        self.enc1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, 32, kernel_size=3, stride=1, dimension=D),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU(),
            ResBlock(32, 32, D),
        )

        self.enc2 = nn.Sequential(
            ME.MinkowskiConvolution(32, 64, kernel_size=3, stride=spatial_stride, dimension=D),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),
            ResBlock(64, 64, D),
        )

        self.enc3 = nn.Sequential(
            ME.MinkowskiConvolution(64, 128, kernel_size=3, stride=spatial_stride, dimension=D),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU(),
            ResBlock(128, 128, D),
        )

        self.enc4 = nn.Sequential(
            ME.MinkowskiConvolution(128, 256, kernel_size=3, stride=spatial_stride, dimension=D),
            ME.MinkowskiBatchNorm(256),
            ME.MinkowskiReLU(),
            ResBlock(256, 256, D),
        )

        # --- Decoder ---
        self.union = ME.MinkowskiUnion()
        self.pruner = ME.MinkowskiPruning()
        self.prune4 = ME.MinkowskiConvolution(128, 1, kernel_size=1, stride=1, dimension=D)
        self.prune3 = ME.MinkowskiConvolution(64, 1, kernel_size=1, stride=1, dimension=D)
        self.prune2 = ME.MinkowskiConvolution(32, 1, kernel_size=1, stride=1, dimension=D)

        self.dec4 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                256, 128, kernel_size=3, stride=spatial_stride, dimension=D, expand_coordinates=True
            ),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU(),
            ResBlock(128, 128, D),
        )

        self.dec3 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                128, 64, kernel_size=3, stride=spatial_stride, dimension=D, expand_coordinates=True
            ),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),
            ResBlock(64, 64, D),
        )

        self.dec2 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                64, 32, kernel_size=3, stride=spatial_stride, dimension=D, expand_coordinates=True
            ),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU(),
            ResBlock(32, 32, D),
        )

        self.dec1 = nn.Sequential(
            ME.MinkowskiConvolution(32, 64, kernel_size=3, stride=1, dimension=D),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),
            ResBlock(64, 64, D),
        )

        self.head_occ = ME.MinkowskiConvolution(64, out_channels, kernel_size=1, stride=1, dimension=D)
        self.head_offset = ME.MinkowskiConvolution(64, 3, kernel_size=1, stride=1, dimension=D)

    def _prune(self, x, prune_layer):
        logits = prune_layer(x).F.squeeze()
        if logits.numel() == 0:
            return x
        mask = logits > self.prune_threshold
        if not torch.any(mask):
            mask[torch.argmax(logits)] = True
        return self.pruner(x, mask)

    def forward(self, x):
        # --- Encoder ---
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # --- Decoder (with Skip Connections) ---
        d4 = self.dec4(e4)
        d4 = self._prune(d4, self.prune4)
        d4_cat = self.union(d4, e3)

        d3 = self.dec3(d4_cat)
        d3 = self._prune(d3, self.prune3)
        d3_cat = self.union(d3, e2)

        d2 = self.dec2(d3_cat)
        d2 = self._prune(d2, self.prune2)
        d2_cat = self.union(d2, e1)

        d1 = self.dec1(d2_cat)
        out_occ = self.head_occ(d1)
        out_off = self.head_offset(d1)
        return out_occ, out_off
