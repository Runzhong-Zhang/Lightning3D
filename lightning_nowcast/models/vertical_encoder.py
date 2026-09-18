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
    """Shared implementation for the current learned vertical encoders."""

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
        self.input_depth = int(input_depth)
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

        self.output_channels = self.out_channels

    def forward(
        self,
        radar: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is not None:
            if mask.shape != radar.shape:
                raise ValueError("Radar and mask must have matching shapes.")
            radar = torch.cat((radar, mask), dim=2)
        if radar.ndim != 6 or radar.shape[2:4] != (self.in_channels, self.input_depth):
            raise ValueError("Expected radar [B,T,C,Z,H,W] with configured channels and depth.")
        b, t, c, z, h, w = radar.shape
        features, _ = self.aggregate(self.layers(radar.reshape(b * t, c, z, h, w)))
        return features.reshape(b, t, self.output_channels, h, w)


class BaselineVerticalEncoder(VerticalEncoder):
    def __init__(self, in_channels: int, input_depth: int):
        super().__init__(in_channels, input_depth)


class CapacityVerticalEncoder(VerticalEncoder):
    def __init__(self, in_channels: int, input_depth: int):
        super().__init__(in_channels, input_depth, out_channels=8)


class MixingVerticalEncoder(VerticalEncoder):
    def __init__(self, in_channels: int, input_depth: int):
        super().__init__(in_channels, input_depth, channel_mixing=True)


class MaxVerticalEncoder(nn.Module):
    def __init__(self, in_channels: int, input_depth: int):
        super().__init__()
        if in_channels != 2:
            raise ValueError("Max encoder requires two combined radar/mask channels.")
        self.input_depth = input_depth
        self.output_channels = 2

    def forward(self, radar: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None or mask.shape != radar.shape:
            raise ValueError("Max encoder requires a matching radar validity mask.")
        if radar.ndim != 6 or radar.shape[2:4] != (1, self.input_depth):
            raise ValueError("Max encoder expects [B,T,1,Z,H,W] with configured depth.")
        valid = mask.bool()
        column_valid = valid.any(dim=3)
        column = radar.masked_fill(~valid, float('-inf')).amax(dim=3)
        column = torch.where(column_valid, column, -1.0)
        return torch.cat((column, column_valid.to(column.dtype)), dim=2)


VERTICAL_ENCODERS = {
    "baseline": BaselineVerticalEncoder,
    "capacity": CapacityVerticalEncoder,
    "mixing": MixingVerticalEncoder,
    "max": MaxVerticalEncoder,
}


def build_vertical_encoder(name: str, input_channels: int, input_depth: int) -> nn.Module:
    """Build the vertical encoder selected in the model configuration."""
    try:
        encoder_class = VERTICAL_ENCODERS[str(name).lower()]
    except KeyError as error:
        available = ", ".join(sorted(VERTICAL_ENCODERS))
        raise ValueError(
            f"Unknown vertical_encoder {name!r}. Available encoders: {available}."
        ) from error
    return encoder_class(int(input_channels), int(input_depth))
