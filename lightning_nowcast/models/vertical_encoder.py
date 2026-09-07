import torch
from torch import nn


class VerticalAttentionAggregator(nn.Module):
    """Content-adaptive, per-variable aggregation over the reduced Z axis."""

    def __init__(self, channels: int, depth: int) -> None:
        super().__init__()
        self.depth = depth
        self.score = nn.Conv3d(
            channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0),
            groups=channels,
        )
        self.level_bias = nn.Parameter(torch.zeros(1, channels, depth, 1, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[2] != self.depth:
            raise ValueError(f"Expected reduced depth {self.depth}, got {x.shape[2]}.")
        weights = torch.softmax(self.score(x) + self.level_bias, dim=2)
        return (x * weights).sum(dim=2), weights


class VerticalEncoder(nn.Module):
    """Vertical-only radar feature extraction followed by depthwise attention."""

    def __init__(
        self,
        in_channels: int,
        input_depth: int,
        num_layers: int = 3,
        out_channels: int | None = None,
        channel_mixing: bool = False,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be positive.")
        if num_layers < 1:
            raise ValueError("num_layers must be positive.")
        self.in_channels = int(in_channels)
        self.out_channels = self.in_channels if out_channels is None else int(out_channels)
        self.channel_mixing = bool(channel_mixing)
        if self.out_channels < 1:
            raise ValueError("out_channels must be positive.")
        if not self.channel_mixing and self.out_channels % self.in_channels != 0:
            raise ValueError(
                "out_channels must be divisible by in_channels when channel_mixing is disabled."
            )

        groups = 1 if self.channel_mixing else self.in_channels
        layer_channels = [self.in_channels] + [self.out_channels] * num_layers
        self.layers = nn.Sequential(*[
            nn.Sequential(
                nn.Conv3d(
                    layer_channels[index], layer_channels[index + 1], kernel_size=(3, 1, 1),
                    stride=(2, 1, 1), padding=(1, 0, 0), groups=groups,
                ),
                nn.GELU(),
            )
            for index in range(num_layers)
        ])
        reduced_depth = input_depth
        for _ in range(num_layers):
            reduced_depth = (reduced_depth + 1) // 2
        self.aggregate = VerticalAttentionAggregator(self.out_channels, reduced_depth)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"Expected (B*T,C,Z,H,W), got {tuple(x.shape)}.")
        features, attention = self.aggregate(self.layers(x))
        return (features, attention) if return_attention else features

    def macs_per_volume(self, depth: int, height: int, width: int) -> int:
        """Approximate Conv3D MACs for one input radar volume."""
        macs = 0
        current_depth = depth
        current_height = height
        current_width = width
        for layer in self.layers:
            conv = layer[0]
            kernel_depth, kernel_height, kernel_width = conv.kernel_size
            stride_depth, stride_height, stride_width = conv.stride
            padding_depth, padding_height, padding_width = conv.padding
            dilation_depth, dilation_height, dilation_width = conv.dilation
            next_depth = (current_depth + 2 * padding_depth - dilation_depth * (kernel_depth - 1) - 1) // stride_depth + 1
            next_height = (current_height + 2 * padding_height - dilation_height * (kernel_height - 1) - 1) // stride_height + 1
            next_width = (current_width + 2 * padding_width - dilation_width * (kernel_width - 1) - 1) // stride_width + 1
            macs += (
                conv.out_channels
                * (conv.in_channels // conv.groups)
                * kernel_depth
                * kernel_height
                * kernel_width
                * next_depth
                * next_height
                * next_width
            )
            current_depth = next_depth
            current_height = next_height
            current_width = next_width

        attention = self.aggregate.score
        kernel_depth, kernel_height, kernel_width = attention.kernel_size
        return macs + (
            attention.out_channels
            * (attention.in_channels // attention.groups)
            * kernel_depth
            * kernel_height
            * kernel_width
            * current_depth
            * current_height
            * current_width
        )
