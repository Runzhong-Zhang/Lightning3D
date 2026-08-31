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
    """Depthwise vertical-only radar feature extraction followed by attention."""

    def __init__(
        self, in_channels: int, input_depth: int, num_layers: int = 3,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive.")
        self.layers = nn.Sequential(*[
            nn.Sequential(
                nn.Conv3d(
                    in_channels, in_channels, kernel_size=(3, 1, 1),
                    stride=(2, 1, 1), padding=(1, 0, 0), groups=in_channels,
                ),
                nn.GELU(),
            )
            for _ in range(num_layers)
        ])
        reduced_depth = input_depth
        for _ in range(num_layers):
            reduced_depth = (reduced_depth + 1) // 2
        self.aggregate = VerticalAttentionAggregator(in_channels, reduced_depth)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"Expected (B*T,C,Z,H,W), got {tuple(x.shape)}.")
        features, attention = self.aggregate(self.layers(x))
        return (features, attention) if return_attention else features

    def macs_per_volume(self, depth: int, height: int, width: int) -> int:
        """Approximate Conv3D MACs for one ``(C, Z, H, W)`` radar volume."""
        macs = 0
        current_depth = depth
        for layer in self.layers:
            next_depth = (current_depth + 1) // 2
            macs += layer[0].in_channels * next_depth * height * width * 3
            current_depth = next_depth
        return macs + self.aggregate.score.in_channels * current_depth * height * width * 3
