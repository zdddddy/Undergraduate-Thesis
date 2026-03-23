import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm(out_channels, norm_type='group', group_norm_groups=8):
    ntype = str(norm_type).lower()
    if ntype == 'batch':
        return nn.BatchNorm2d(out_channels)
    if ntype == 'instance':
        return nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=False)
    if ntype == 'group':
        groups = int(max(1, group_norm_groups))
        groups = min(groups, out_channels)
        while out_channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, out_channels)
    raise ValueError(f'Unsupported norm_type={norm_type}, expected batch/group/instance')


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type='group', group_norm_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm(out_channels, norm_type=norm_type, group_norm_groups=group_norm_groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm(out_channels, norm_type=norm_type, group_norm_groups=group_norm_groups),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, norm_type='group', group_norm_groups=8):
        super().__init__()
        self.conv = ConvBlock(
            in_channels + skip_channels,
            out_channels,
            norm_type=norm_type,
            group_norm_groups=group_norm_groups,
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class HeightRecurrentUNet(nn.Module):
    def __init__(
        self,
        in_channels=4,
        base_channels=32,
        out_channels=1,
        use_edge_head=False,
        norm_type='group',
        group_norm_groups=8,
    ):
        super().__init__()
        self.use_edge_head = bool(use_edge_head)
        self.enc1 = ConvBlock(
            in_channels,
            base_channels,
            norm_type=norm_type,
            group_norm_groups=group_norm_groups,
        )
        self.enc2 = ConvBlock(
            base_channels,
            base_channels * 2,
            norm_type=norm_type,
            group_norm_groups=group_norm_groups,
        )
        self.enc3 = ConvBlock(
            base_channels * 2,
            base_channels * 4,
            norm_type=norm_type,
            group_norm_groups=group_norm_groups,
        )
        self.bottleneck = ConvBlock(
            base_channels * 4,
            base_channels * 8,
            norm_type=norm_type,
            group_norm_groups=group_norm_groups,
        )

        self.up3 = UpBlock(
            base_channels * 8,
            base_channels * 4,
            base_channels * 4,
            norm_type=norm_type,
            group_norm_groups=group_norm_groups,
        )
        self.up2 = UpBlock(
            base_channels * 4,
            base_channels * 2,
            base_channels * 2,
            norm_type=norm_type,
            group_norm_groups=group_norm_groups,
        )
        self.up1 = UpBlock(
            base_channels * 2,
            base_channels,
            base_channels,
            norm_type=norm_type,
            group_norm_groups=group_norm_groups,
        )
        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        self.edge_head = nn.Conv2d(base_channels, 1, kernel_size=1) if self.use_edge_head else None

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, kernel_size=2, stride=2))
        e3 = self.enc3(F.max_pool2d(e2, kernel_size=2, stride=2))
        b = self.bottleneck(F.max_pool2d(e3, kernel_size=2, stride=2))

        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        h = self.head(d1)
        if self.use_edge_head:
            e = self.edge_head(d1)
            return h, e
        return h


# Backward-compatible alias.
RecurrentUNet = HeightRecurrentUNet
