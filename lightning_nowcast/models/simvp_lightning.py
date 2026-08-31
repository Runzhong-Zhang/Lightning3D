from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lightning_nowcast.models.vertical_encoder import VerticalEncoder


def _shape_str(x: torch.Tensor) -> str:
    return "x".join(str(v) for v in x.shape)


def _safe_logit_from_probability(prob: torch.Tensor, eps: float = 1.0e-4) -> torch.Tensor:
    prob = prob.clamp(min=float(eps), max=1.0 - float(eps))
    return torch.logit(prob, eps=float(eps))


class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        upsampling: bool = False,
        act_norm: bool = False,
        act_inplace: bool = True,
    ):
        super().__init__()
        self.act_norm = act_norm
        if upsampling:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * 4,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                ),
                nn.PixelShuffle(2),
            )
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )

        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.SiLU(inplace=act_inplace)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Conv2d):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        downsampling: bool = False,
        upsampling: bool = False,
        act_norm: bool = True,
        act_inplace: bool = True,
    ):
        super().__init__()
        stride = 2 if downsampling else 1
        padding = (kernel_size - stride + 1) // 2
        self.conv = BasicConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            upsampling=upsampling,
            act_norm=act_norm,
            act_inplace=act_inplace,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class GroupConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        act_norm: bool = False,
        act_inplace: bool = True,
    ):
        super().__init__()
        if in_channels % groups != 0:
            groups = 1
        self.act_norm = act_norm
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
        )
        self.norm = nn.GroupNorm(groups, out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=act_inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y


class GInceptionST(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        inception_kernels: tuple[int, ...] = (3, 5, 7, 11),
        groups: int = 8,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.layers = nn.ModuleList(
            [
                GroupConv2d(
                    hidden_channels,
                    out_channels,
                    kernel_size=kernel,
                    stride=1,
                    padding=kernel // 2,
                    groups=groups,
                    act_norm=True,
                )
                for kernel in inception_kernels
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y = y + layer(x)
        return y


def _sampling_generator(num_stages: int, reverse: bool = False, enabled: bool = True) -> list[bool]:
    if not enabled:
        samplings = [False] * num_stages
    else:
        samplings = [False, True] * (num_stages // 2)
        samplings = samplings[:num_stages]
    return list(reversed(samplings)) if reverse else samplings


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_stages: int,
        spatio_kernel: int,
        use_spatial_sampling: bool = False,
        act_inplace: bool = True,
    ):
        super().__init__()
        samplings = _sampling_generator(num_stages, enabled=use_spatial_sampling)
        layers = [
            ConvSC(
                in_channels,
                hidden_channels,
                spatio_kernel,
                downsampling=samplings[0],
                act_inplace=act_inplace,
            )
        ]
        for sampling in samplings[1:]:
            layers.append(
                ConvSC(
                    hidden_channels,
                    hidden_channels,
                    spatio_kernel,
                    downsampling=sampling,
                    act_inplace=act_inplace,
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        enc1 = self.layers[0](x)
        latent = enc1
        for layer in self.layers[1:]:
            latent = layer(latent)
        return latent, enc1


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        num_stages: int,
        spatio_kernel: int,
        use_spatial_sampling: bool = False,
        act_inplace: bool = True,
    ):
        super().__init__()
        samplings = _sampling_generator(num_stages, reverse=True, enabled=use_spatial_sampling)
        layers = []
        for sampling in samplings[:-1]:
            layers.append(
                ConvSC(
                    hidden_channels,
                    hidden_channels,
                    spatio_kernel,
                    upsampling=sampling,
                    act_inplace=act_inplace,
                )
            )
        layers.append(
            ConvSC(
                hidden_channels,
                hidden_channels,
                spatio_kernel,
                upsampling=samplings[-1],
                act_inplace=act_inplace,
            )
        )
        self.layers = nn.ModuleList(layers)
        self.readout = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, hidden: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            hidden = layer(hidden)
        y = self.layers[-1](hidden + skip)
        return self.readout(y)


class MidIncepNet(nn.Module):
    def __init__(
        self,
        channel_in: int,
        channel_hid: int,
        num_blocks: int,
        inception_kernels: tuple[int, ...] = (3, 5, 7, 11),
        groups: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_blocks < 2:
            raise ValueError("SimVP translator needs at least 2 blocks.")
        self.num_blocks = num_blocks
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        enc_layers = [
            GInceptionST(
                channel_in,
                channel_hid // 2,
                channel_hid,
                inception_kernels=inception_kernels,
                groups=groups,
            )
        ]
        for _ in range(1, num_blocks - 1):
            enc_layers.append(
                GInceptionST(
                    channel_hid,
                    channel_hid // 2,
                    channel_hid,
                    inception_kernels=inception_kernels,
                    groups=groups,
                )
            )
        enc_layers.append(
            GInceptionST(
                channel_hid,
                channel_hid // 2,
                channel_hid,
                inception_kernels=inception_kernels,
                groups=groups,
            )
        )

        dec_layers = [
            GInceptionST(
                channel_hid,
                channel_hid // 2,
                channel_hid,
                inception_kernels=inception_kernels,
                groups=groups,
            )
        ]
        for _ in range(1, num_blocks - 1):
            dec_layers.append(
                GInceptionST(
                    2 * channel_hid,
                    channel_hid // 2,
                    channel_hid,
                    inception_kernels=inception_kernels,
                    groups=groups,
                )
            )
        dec_layers.append(
            GInceptionST(
                2 * channel_hid,
                channel_hid // 2,
                channel_in,
                inception_kernels=inception_kernels,
                groups=groups,
            )
        )

        self.enc = nn.ModuleList(enc_layers)
        self.dec = nn.ModuleList(dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, channels, height, width = x.shape
        z = x.reshape(batch_size, time_steps * channels, height, width)

        skips = []
        for block_idx, block in enumerate(self.enc):
            z = block(z)
            z = self.drop(z)
            if block_idx < self.num_blocks - 1:
                skips.append(z)

        z = self.dec[0](z)
        z = self.drop(z)
        for block_idx in range(1, self.num_blocks):
            z = self.dec[block_idx](torch.cat([z, skips[-block_idx]], dim=1))
            z = self.drop(z)

        return z.reshape(batch_size, time_steps, channels, height, width)


class SimVPBackbone(nn.Module):
    def __init__(
        self,
        input_steps: int,
        output_steps: int,
        in_channels: int,
        out_channels: int,
        input_hw: tuple[int, int],
        hid_S: int = 32,
        hid_T: int = 256,
        N_S: int = 4,
        N_T: int = 4,
        spatio_kernel_enc: int = 3,
        spatio_kernel_dec: int = 3,
        use_spatial_sampling: bool = False,
        act_inplace: bool = False,
        dropout: float = 0.0,
        use_lead_time_emb: bool = False,
    ):
        super().__init__()
        self.input_steps = int(input_steps)
        self.output_steps = int(output_steps)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.input_hw = tuple(int(v) for v in input_hw)
        self.use_lead_time_emb = bool(use_lead_time_emb)

        if use_spatial_sampling:
            reduced_h = int(input_hw[0] / 2 ** (N_S / 2))
            reduced_w = int(input_hw[1] / 2 ** (N_S / 2))
        else:
            reduced_h, reduced_w = int(input_hw[0]), int(input_hw[1])

        self.encoder = Encoder(
            in_channels,
            hid_S,
            N_S,
            spatio_kernel_enc,
            use_spatial_sampling=use_spatial_sampling,
            act_inplace=act_inplace,
        )
        self.decoder = Decoder(
            hid_S,
            out_channels,
            N_S,
            spatio_kernel_dec,
            use_spatial_sampling=use_spatial_sampling,
            act_inplace=act_inplace,
        )
        self.translator = MidIncepNet(
            channel_in=self.input_steps * hid_S,
            channel_hid=hid_T,
            num_blocks=N_T,
            dropout=dropout,
        )
        self.temporal_proj = None
        if self.output_steps != self.input_steps:
            self.temporal_proj = nn.Linear(self.input_steps, self.output_steps)

        self.lead_time_emb = None
        if self.use_lead_time_emb:
            self.lead_time_emb = nn.Parameter(torch.zeros(self.output_steps, hid_S, 1, 1))

        self.reduced_hw = (reduced_h, reduced_w)

    def encode_translate(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(
                "SimVP backbone expects channel-first video input with shape "
                f"(B, T, C, H, W), got {_shape_str(x)}"
            )
        batch_size, time_steps, channels, height, width = x.shape
        if time_steps != self.input_steps:
            raise ValueError(f"Expected {self.input_steps} input steps, got {time_steps}")
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {channels}")

        x_flat = x.reshape(batch_size * time_steps, channels, height, width)
        embed, skip = self.encoder(x_flat)
        _, hidden_channels, hidden_h, hidden_w = embed.shape

        latent = embed.reshape(batch_size, time_steps, hidden_channels, hidden_h, hidden_w)
        latent = self.translator(latent)

        if self.temporal_proj is not None:
            latent = latent.permute(0, 3, 4, 2, 1).contiguous()
            latent = latent.reshape(-1, self.input_steps)
            latent = self.temporal_proj(latent)
            latent = latent.reshape(batch_size, hidden_h, hidden_w, hidden_channels, self.output_steps)
            latent = latent.permute(0, 4, 3, 1, 2).contiguous()

        if self.lead_time_emb is not None:
            latent = latent + self.lead_time_emb.unsqueeze(0)

        out_steps = latent.shape[1]
        latent_flat = latent.reshape(batch_size * out_steps, hidden_channels, hidden_h, hidden_w)

        if out_steps == time_steps:
            skip_used = skip
        else:
            skip_5d = skip.reshape(batch_size, time_steps, hidden_channels, height, width)
            skip_5d = skip_5d.permute(0, 3, 4, 2, 1).contiguous()
            skip_5d = skip_5d.reshape(-1, self.input_steps)
            skip_5d = self.temporal_proj(skip_5d)
            skip_5d = skip_5d.reshape(batch_size, height, width, hidden_channels, out_steps)
            skip_5d = skip_5d.permute(0, 4, 3, 1, 2).contiguous()
            skip_used = skip_5d.reshape(batch_size * out_steps, hidden_channels, height, width)

        return latent, skip_used

    def decode_from_latent(self, latent: torch.Tensor, skip_used: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 5:
            raise ValueError(
                "SimVP decode_from_latent expects latent shape (B, T, C, H, W), "
                f"got {_shape_str(latent)}"
            )

        batch_size, out_steps, hidden_channels, hidden_h, hidden_w = latent.shape
        latent_flat = latent.reshape(batch_size * out_steps, hidden_channels, hidden_h, hidden_w)
        height, width = self.input_hw

        decoded = self.decoder(latent_flat, skip_used)
        decoded = decoded.reshape(batch_size, out_steps, self.out_channels, height, width)
        expected_shape = (batch_size, self.output_steps, self.out_channels, height, width)
        if tuple(decoded.shape) != expected_shape:
            raise RuntimeError(
                f"Unexpected SimVP decoded shape {tuple(decoded.shape)}, expected {expected_shape}"
            )
        return decoded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent, skip_used = self.encode_translate(x)
        batch_size, _, _, height, width = x.shape
        return self.decode_from_latent(latent, skip_used)


class LightningSimVPSystem(nn.Module):
    def __init__(
        self,
        lag_time: int,
        lead_time: int,
        target_hw: tuple[int, int],
        radar_input_hw: tuple[int, int],
        hid_S: int = 32,
        hid_T: int = 256,
        N_S: int = 4,
        N_T: int = 4,
        spatio_kernel_enc: int = 3,
        spatio_kernel_dec: int = 3,
        dropout: float = 0.0,
        use_spatial_sampling: bool = True,
        predict_uncertainty: bool = False,
        use_future_condition: bool = False,
        future_condition_channels: int = 2,
        future_condition_mode: str = "simple",
        input_mode: str = "two_stream",
        lightning_input_channels: int = 1,
        radar_input_channels: int = 1,
        radar_input_layout: str = "channel_last",
        radar_vertical_levels: int | None = None,
        vertical_encoder_layers: int = 3,
        uncertainty_eps: float = 1.0e-4,
        use_lead_time_emb: bool = False,
        use_time_emb: bool = False,
    ):
        super().__init__()
        self.lag_time = int(lag_time)
        self.lead_time = int(lead_time)
        self.target_hw = tuple(int(v) for v in target_hw)
        self.radar_input_hw = tuple(int(v) for v in radar_input_hw)
        self.predict_uncertainty = bool(predict_uncertainty)
        self.use_future_condition = bool(use_future_condition)
        self.future_condition_channels = int(future_condition_channels)
        self.future_condition_mode = str(future_condition_mode).lower()
        self.input_mode = str(input_mode).lower()
        self.lightning_input_channels = int(lightning_input_channels)
        self.radar_input_channels = int(radar_input_channels)
        self.radar_input_layout = str(radar_input_layout).lower()
        self.radar_vertical_levels = None if radar_vertical_levels is None else int(radar_vertical_levels)
        self.uncertainty_eps = float(uncertainty_eps)
        self.final_output_channels = 2 if self.predict_uncertainty else 1
        self.stem_channels = int(hid_S)
        self.use_time_emb = bool(use_time_emb)
        self.time_emb_proj = nn.Linear(4, self.stem_channels) if self.use_time_emb else None
        self.predecoder_future_fusion = self.use_future_condition and self.future_condition_mode == "latent_predecoder"
        if self.input_mode not in {"two_stream", "stacked"}:
            raise ValueError(f"Unsupported input_mode {self.input_mode!r}. Expected 'two_stream' or 'stacked'.")
        if self.lightning_input_channels < 1:
            raise ValueError("lightning_input_channels must be >= 1.")
        if self.radar_input_channels < 1:
            raise ValueError("radar_input_channels must be >= 1.")
        if self.radar_input_layout not in {"channel_last", "channel_first", "channel_first_3d"}:
            raise ValueError(
                "radar_input_layout must be 'channel_last', 'channel_first', or 'channel_first_3d'."
            )
        if self.radar_input_layout == "channel_first_3d":
            if self.radar_vertical_levels is None or self.radar_vertical_levels < 1:
                raise ValueError("channel_first_3d radar inputs require radar_vertical_levels >= 1.")
            self.vertical_encoder = VerticalEncoder(
                in_channels=self.radar_input_channels,
                input_depth=self.radar_vertical_levels,
                num_layers=int(vertical_encoder_layers),
            )
        else:
            self.vertical_encoder = None
        if self.future_condition_mode not in {"simple", "interval_attention", "latent_predecoder"}:
            raise ValueError(
                "Unsupported future_condition_mode "
                f"{self.future_condition_mode!r}. Expected 'simple', 'interval_attention', or 'latent_predecoder'."
            )

        if self.input_mode == "two_stream":
            self.radar_stem = nn.Sequential(
                BasicConv2d(
                    self.radar_input_channels,
                    self.stem_channels,
                    kernel_size=3,
                    padding=1,
                    act_norm=True,
                    act_inplace=False,
                ),
                BasicConv2d(
                    self.stem_channels,
                    self.stem_channels,
                    kernel_size=3,
                    padding=1,
                    act_norm=True,
                    act_inplace=False,
                ),
            )
            self.lightning_stem = nn.Sequential(
                BasicConv2d(
                    self.lightning_input_channels,
                    self.stem_channels,
                    kernel_size=3,
                    padding=1,
                    act_norm=True,
                    act_inplace=False,
                ),
                BasicConv2d(
                    self.stem_channels,
                    self.stem_channels,
                    kernel_size=3,
                    padding=1,
                    act_norm=True,
                    act_inplace=False,
                ),
            )
            self.fusion_refine = nn.Sequential(
                BasicConv2d(
                    2 * self.stem_channels,
                    self.stem_channels,
                    kernel_size=3,
                    padding=1,
                    act_norm=True,
                    act_inplace=False,
                ),
                BasicConv2d(
                    self.stem_channels,
                    self.stem_channels,
                    kernel_size=3,
                    padding=1,
                    act_norm=True,
                    act_inplace=False,
                ),
                BasicConv2d(
                    self.stem_channels,
                    self.stem_channels,
                    kernel_size=3,
                    padding=1,
                    act_norm=True,
                    act_inplace=False,
                ),
            )
            self.fusion_drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
            backbone_in_channels = self.stem_channels
        else:
            self.radar_stem = None
            self.lightning_stem = None
            self.fusion_refine = None
            self.fusion_drop = nn.Identity()
            backbone_in_channels = self.radar_input_channels + self.lightning_input_channels
        self.backbone_feature_channels = (
            self.stem_channels if self.use_future_condition and not self.predecoder_future_fusion else self.final_output_channels
        )
        self.backbone = SimVPBackbone(
            input_steps=self.lag_time,
            output_steps=self.lead_time,
            in_channels=backbone_in_channels,
            out_channels=self.backbone_feature_channels,
            input_hw=self.target_hw,
            hid_S=self.stem_channels,
            hid_T=hid_T,
            N_S=N_S,
            N_T=N_T,
            spatio_kernel_enc=spatio_kernel_enc,
            spatio_kernel_dec=spatio_kernel_dec,
            use_spatial_sampling=use_spatial_sampling,
            act_inplace=False,
            dropout=dropout,
            use_lead_time_emb=use_lead_time_emb,
        )

        if self.use_future_condition:
            if self.future_condition_mode == "simple":
                self.future_condition_adapter = nn.Sequential(
                    BasicConv2d(
                        self.future_condition_channels,
                        self.stem_channels,
                        kernel_size=3,
                        padding=1,
                        act_norm=True,
                        act_inplace=False,
                    ),
                    BasicConv2d(
                        self.stem_channels,
                        self.stem_channels,
                        kernel_size=3,
                        padding=1,
                        act_norm=True,
                        act_inplace=False,
                    ),
                )
                self.future_condition_branch_adapter = None
                self.future_condition_spread_adapter = None
                self.future_condition_branch_attention = None
                self.latent_condition_fusion = None
                self.latent_condition_drop = nn.Identity()
            elif self.future_condition_mode == "interval_attention":
                if self.future_condition_channels != 2:
                    raise ValueError(
                        "future_condition_mode='interval_attention' requires future_condition_channels=2 "
                        "for mean and uncertainty."
                    )
                self.future_condition_adapter = None
                self.future_condition_branch_adapter = nn.Sequential(
                    BasicConv2d(
                        1,
                        self.stem_channels,
                        kernel_size=3,
                        padding=1,
                        act_norm=True,
                        act_inplace=False,
                    ),
                    BasicConv2d(
                        self.stem_channels,
                        self.stem_channels,
                        kernel_size=3,
                        padding=1,
                        act_norm=True,
                        act_inplace=False,
                    ),
                )
                self.future_condition_spread_adapter = nn.Sequential(
                    BasicConv2d(
                        1,
                        self.stem_channels,
                        kernel_size=3,
                        padding=1,
                        act_norm=True,
                        act_inplace=False,
                    ),
                    nn.Conv2d(self.stem_channels, 1, kernel_size=1),
                )
                self.future_condition_branch_attention = nn.Sequential(
                    BasicConv2d(
                        2 * self.stem_channels,
                        self.stem_channels,
                        kernel_size=3,
                        padding=1,
                        act_norm=True,
                        act_inplace=False,
                    ),
                    BasicConv2d(
                        self.stem_channels,
                        self.stem_channels,
                        kernel_size=3,
                        padding=1,
                        act_norm=True,
                        act_inplace=False,
                    ),
                    nn.Conv2d(self.stem_channels, 1, kernel_size=1),
                )
                self.latent_condition_fusion = None
                self.latent_condition_drop = nn.Identity()
            else:
                self.future_condition_adapter = nn.Sequential(
                    BasicConv2d(
                        self.future_condition_channels,
                        self.stem_channels,
                        kernel_size=3,
                        padding=1,
                        act_norm=True,
                        act_inplace=False,
                    ),
                    BasicConv2d(
                        self.stem_channels,
                        self.stem_channels,
                        kernel_size=3,
                        padding=1,
                        act_norm=True,
                        act_inplace=False,
                    ),
                )
                self.future_condition_branch_adapter = None
                self.future_condition_spread_adapter = None
                self.future_condition_branch_attention = None
                self.latent_condition_fusion = nn.Sequential(
                    BasicConv2d(
                        2 * self.stem_channels,
                        self.stem_channels,
                        kernel_size=3,
                        padding=1,
                        act_norm=True,
                        act_inplace=False,
                    ),
                    BasicConv2d(
                        self.stem_channels,
                        self.stem_channels,
                        kernel_size=3,
                        padding=1,
                        act_norm=True,
                        act_inplace=False,
                    ),
                )
                self.latent_condition_drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

            if self.predecoder_future_fusion:
                self.output_fusion = None
                self.output_head = None
                self.output_condition_drop = nn.Identity()
            else:
                self.output_fusion = nn.Sequential(
                    BasicConv2d(
                        2 * self.stem_channels,
                        self.stem_channels,
                        kernel_size=3,
                        padding=1,
                        act_norm=True,
                        act_inplace=False,
                    ),
                    BasicConv2d(
                        self.stem_channels,
                        self.stem_channels,
                        kernel_size=3,
                        padding=1,
                        act_norm=True,
                        act_inplace=False,
                    ),
                )
                self.output_head = nn.Conv2d(self.stem_channels, self.final_output_channels, kernel_size=1)
                self.output_condition_drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        else:
            self.future_condition_adapter = None
            self.future_condition_branch_adapter = None
            self.future_condition_spread_adapter = None
            self.future_condition_branch_attention = None
            self.latent_condition_fusion = None
            self.latent_condition_drop = nn.Identity()
            self.output_fusion = None
            self.output_head = None
            self.output_condition_drop = nn.Identity()

    def prepare_inputs(self, radar_past: torch.Tensor, lightning_past: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expected_radar_ndim = 6 if self.radar_input_layout == "channel_first_3d" else 5
        if radar_past.ndim != expected_radar_ndim:
            raise ValueError(
                f"Expected {self.radar_input_layout} radar_past with {expected_radar_ndim} dimensions, "
                f"got {_shape_str(radar_past)}"
            )
        if lightning_past.ndim != 5:
            raise ValueError(
                "Expected lightning_past with shape (B, T, H, W, C), "
                f"got {_shape_str(lightning_past)}"
            )

        if self.radar_input_layout == "channel_last":
            batch_size, time_steps, radar_h, radar_w, radar_channels = radar_past.shape
            radar = radar_past.permute(0, 1, 4, 2, 3).contiguous()
        elif self.radar_input_layout == "channel_first_3d":
            batch_size, time_steps, radar_channels, radar_depth, radar_h, radar_w = radar_past.shape
            if radar_depth != self.radar_vertical_levels:
                raise ValueError(
                    f"Expected radar vertical levels={self.radar_vertical_levels}, got {radar_depth}"
                )
            assert self.vertical_encoder is not None
            radar = self.vertical_encoder(
                radar_past.reshape(batch_size * time_steps, radar_channels, radar_depth, radar_h, radar_w)
            ).reshape(batch_size, time_steps, radar_channels, radar_h, radar_w)
        else:
            batch_size, time_steps, radar_channels, radar_h, radar_w = radar_past.shape
            radar = radar_past.contiguous()
        lgt_batch, lgt_steps, lgt_h, lgt_w, lgt_channels = lightning_past.shape
        if batch_size != lgt_batch or time_steps != lgt_steps:
            raise ValueError("Radar and lightning inputs must share batch and time dimensions.")
        if time_steps != self.lag_time:
            raise ValueError(f"Expected lag_time={self.lag_time}, got {time_steps}")
        if radar_channels != self.radar_input_channels:
            raise ValueError(f"Expected radar input channels={self.radar_input_channels}, got {radar_channels}")
        if lgt_channels != self.lightning_input_channels:
            raise ValueError(
                f"Expected lightning input channels={self.lightning_input_channels}, got {lgt_channels}"
            )
        if (radar_h, radar_w) != self.radar_input_hw:
            raise ValueError(f"Expected radar size {self.radar_input_hw}, got {(radar_h, radar_w)}")
        if (lgt_h, lgt_w) != self.target_hw:
            raise ValueError(f"Expected lightning size {self.target_hw}, got {(lgt_h, lgt_w)}")

        radar = radar.reshape(batch_size * time_steps, self.radar_input_channels, radar_h, radar_w)
        if (radar_h, radar_w) != self.target_hw:
            radar = F.interpolate(radar, size=self.target_hw, mode="bilinear", align_corners=False)
        radar = radar.reshape(batch_size, time_steps, self.radar_input_channels, self.target_hw[0], self.target_hw[1])
        lightning = lightning_past.permute(0, 1, 4, 2, 3).contiguous()
        return radar, lightning

    def fuse_inputs(self, radar: torch.Tensor, lightning: torch.Tensor) -> torch.Tensor:
        if radar.shape != lightning.shape:
            if radar.shape[:2] != lightning.shape[:2] or radar.shape[3:] != lightning.shape[3:]:
                raise ValueError(
                    f"Aligned modality tensors must share batch/time/spatial shape, got radar={tuple(radar.shape)} vs lightning={tuple(lightning.shape)}"
                )
        batch_size, time_steps, _, height, width = radar.shape
        if self.input_mode == "stacked":
            return torch.cat([radar, lightning], dim=2)

        radar_flat = radar.reshape(batch_size * time_steps, self.radar_input_channels, height, width)
        lightning_flat = lightning.reshape(batch_size * time_steps, self.lightning_input_channels, height, width)
        radar_features = self.radar_stem(radar_flat)
        lightning_features = self.lightning_stem(lightning_flat)
        fused_flat = torch.cat([radar_features, lightning_features], dim=1)
        fused_flat = self.fusion_refine(fused_flat)
        fused_flat = self.fusion_drop(fused_flat)
        return fused_flat.reshape(batch_size, time_steps, self.stem_channels, height, width)

    def prepare_future_condition(self, future_condition: torch.Tensor, batch_size: int) -> dict[str, torch.Tensor]:
        if future_condition is None:
            raise ValueError("future_condition must be provided when use_future_condition=True.")
        if future_condition.ndim != 5:
            raise ValueError(
                "Expected future_condition with shape (B, T_out, H, W, C), "
                f"got {_shape_str(future_condition)}"
            )
        if future_condition.shape[0] != batch_size:
            raise ValueError("Batch mismatch between lightning inputs and future condition.")
        if future_condition.shape[1] != self.lead_time:
            raise ValueError(f"Expected lead_time={self.lead_time}, got {future_condition.shape[1]}")
        condition_hw = self.backbone.reduced_hw if self.predecoder_future_fusion else self.target_hw
        if tuple(future_condition.shape[2:4]) != condition_hw:
            raise ValueError(
                f"Expected future condition spatial size {condition_hw}, got {tuple(future_condition.shape[2:4])}"
            )
        if future_condition.shape[4] != self.future_condition_channels:
            raise ValueError(
                f"Expected future condition channels={self.future_condition_channels}, got {future_condition.shape[4]}"
            )
        cond = future_condition.permute(0, 1, 4, 2, 3).contiguous()
        cond = cond.reshape(
            batch_size * self.lead_time,
            self.future_condition_channels,
            condition_hw[0],
            condition_hw[1],
        )
        if self.future_condition_mode in {"simple", "latent_predecoder"}:
            if self.future_condition_adapter is None:
                raise RuntimeError("Future-condition adapter is not initialized.")
            cond = self.future_condition_adapter(cond)
            cond = self.latent_condition_drop(cond) if self.predecoder_future_fusion else self.output_condition_drop(cond)
            return {
                "condition_features": cond.reshape(
                    batch_size,
                    self.lead_time,
                    self.stem_channels,
                    condition_hw[0],
                    condition_hw[1],
                )
            }

        if self.future_condition_branch_adapter is None or self.future_condition_spread_adapter is None:
            raise RuntimeError("Interval-attention future-condition modules are not initialized.")
        mean = cond[:, 0:1]
        uncertainty = cond[:, 1:2]
        spread = F.softplus(self.future_condition_spread_adapter(uncertainty))
        branch_inputs = torch.stack([mean - spread, mean, mean + spread], dim=1)
        return {
            "branch_inputs": branch_inputs.reshape(batch_size, self.lead_time, 3, 1, self.target_hw[0], self.target_hw[1]),
            "branch_spread": spread.reshape(batch_size, self.lead_time, 1, self.target_hw[0], self.target_hw[1]),
        }

    def condition_latent(
        self,
        backbone_latent: torch.Tensor,
        future_condition: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        if self.latent_condition_fusion is None:
            raise RuntimeError("Latent future-condition fusion is not initialized.")
        batch_size = backbone_latent.shape[0]
        prepared = self.prepare_future_condition(future_condition, batch_size=batch_size)
        cond = prepared["condition_features"]
        if backbone_latent.shape != cond.shape:
            raise RuntimeError(
                "Backbone latent / future feature shape mismatch: "
                f"backbone={tuple(backbone_latent.shape)} vs cond={tuple(cond.shape)}"
            )
        reduced_h, reduced_w = self.backbone.reduced_hw
        fused = torch.cat([backbone_latent, cond], dim=2)
        fused = fused.reshape(batch_size * self.lead_time, 2 * self.stem_channels, reduced_h, reduced_w)
        fused = self.latent_condition_fusion(fused)
        fused = self.latent_condition_drop(fused)
        fused_latent = fused.reshape(batch_size, self.lead_time, self.stem_channels, reduced_h, reduced_w)
        return {
            "condition_features": cond,
            "fused_latent": fused_latent,
        }

    def condition_output(
        self,
        backbone_output: torch.Tensor,
        future_condition: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        if self.output_fusion is None or self.output_head is None:
            raise RuntimeError("Future-conditioned output layers are not initialized.")
        batch_size = backbone_output.shape[0]
        prepared = self.prepare_future_condition(future_condition, batch_size=batch_size)

        if self.future_condition_mode == "simple":
            cond = prepared["condition_features"]
            if backbone_output.shape != cond.shape:
                raise RuntimeError(
                    "Backbone output / future condition shape mismatch: "
                    f"backbone={tuple(backbone_output.shape)} vs cond={tuple(cond.shape)}"
                )
            fused = torch.cat([backbone_output, cond], dim=2)
            branch_outputs: dict[str, torch.Tensor] = {}
        else:
            if self.future_condition_branch_attention is None:
                raise RuntimeError("Interval-attention branch scorer is not initialized.")
            branch_inputs = prepared["branch_inputs"]
            expected_branch_shape = (
                batch_size,
                self.lead_time,
                3,
                1,
                self.target_hw[0],
                self.target_hw[1],
            )
            if tuple(branch_inputs.shape) != expected_branch_shape:
                raise RuntimeError(
                    "Unexpected interval branch shape: "
                    f"got {tuple(branch_inputs.shape)}, expected {expected_branch_shape}"
                )
            backbone_flat = backbone_output.reshape(
                batch_size * self.lead_time,
                self.stem_channels,
                self.target_hw[0],
                self.target_hw[1],
            )
            branch_inputs_flat = branch_inputs.reshape(
                batch_size * self.lead_time,
                3,
                1,
                self.target_hw[0],
                self.target_hw[1],
            )
            attention_logits = []
            for branch_idx in range(3):
                branch_feature = self.future_condition_branch_adapter(branch_inputs_flat[:, branch_idx])
                attention_input = torch.cat([backbone_flat, branch_feature], dim=1)
                attention_logits.append(self.future_condition_branch_attention(attention_input))
            branch_attention_logits = torch.cat(attention_logits, dim=1)
            branch_attention = torch.softmax(branch_attention_logits, dim=1)
            cond_flat = torch.zeros_like(backbone_flat)
            for branch_idx in range(3):
                branch_feature = self.future_condition_branch_adapter(branch_inputs_flat[:, branch_idx])
                cond_flat = cond_flat + branch_feature * branch_attention[:, branch_idx:branch_idx + 1]
            cond = cond_flat.reshape(
                batch_size,
                self.lead_time,
                self.stem_channels,
                self.target_hw[0],
                self.target_hw[1],
            )
            fused = torch.cat([backbone_output, cond], dim=2)
            branch_outputs = {
                "branch_attention": branch_attention.reshape(
                    batch_size,
                    self.lead_time,
                    3,
                    self.target_hw[0],
                    self.target_hw[1],
                ),
                "branch_attention_logits": branch_attention_logits.reshape(
                    batch_size,
                    self.lead_time,
                    3,
                    self.target_hw[0],
                    self.target_hw[1],
                ),
                "branch_inputs": prepared["branch_inputs"],
                "branch_spread": prepared["branch_spread"],
            }

        fused = fused.reshape(batch_size * self.lead_time, 2 * self.stem_channels, self.target_hw[0], self.target_hw[1])
        fused = self.output_fusion(fused)
        fused = self.output_condition_drop(fused)
        raw_output = self.output_head(fused)
        branch_outputs["raw_output"] = raw_output.reshape(
            batch_size,
            self.lead_time,
            self.final_output_channels,
            self.target_hw[0],
            self.target_hw[1],
        )
        return branch_outputs

    def _apply_time_emb(self, latent: torch.Tensor, time_emb: torch.Tensor | None) -> torch.Tensor:
        if self.time_emb_proj is None or time_emb is None:
            return latent
        te = self.time_emb_proj(time_emb)  # (B, hid_S)
        return latent + te[:, None, :, None, None]  # broadcast over T, H, W

    def predict_distribution(
        self,
        radar_past: torch.Tensor,
        lightning_past: torch.Tensor,
        future_condition: torch.Tensor | None = None,
        time_emb: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        radar, lightning = self.prepare_inputs(radar_past, lightning_past)
        inputs = self.fuse_inputs(radar, lightning)
        if self.predecoder_future_fusion:
            backbone_latent, skip_used = self.backbone.encode_translate(inputs)
            backbone_latent = self._apply_time_emb(backbone_latent, time_emb)
            condition_outputs = self.condition_latent(backbone_latent, future_condition)
            raw_output = self.backbone.decode_from_latent(condition_outputs["fused_latent"], skip_used)
        else:
            if self.time_emb_proj is not None and time_emb is not None:
                backbone_latent, skip_used = self.backbone.encode_translate(inputs)
                backbone_latent = self._apply_time_emb(backbone_latent, time_emb)
                backbone_output = self.backbone.decode_from_latent(backbone_latent, skip_used)
            else:
                backbone_output = self.backbone(inputs)
            condition_outputs = self.condition_output(backbone_output, future_condition) if self.use_future_condition else {}
            raw_output = condition_outputs.get("raw_output", backbone_output)

        if self.predict_uncertainty:
            alpha = F.softplus(raw_output) + 1.0
            strength = alpha.sum(dim=2, keepdim=True).clamp_min(self.uncertainty_eps)
            probs = alpha[:, :, 1:2] / strength
            logits = _safe_logit_from_probability(probs, eps=self.uncertainty_eps)
            uncertainty = 2.0 / strength
            variance = alpha[:, :, 1:2] * (strength - alpha[:, :, 1:2]) / (
                strength * strength * (strength + 1.0).clamp_min(self.uncertainty_eps)
            )
            outputs = {
                "raw_output": raw_output,
                "logits": logits,
                "probs": probs,
                "alpha": alpha,
                "strength": strength,
                "uncertainty": uncertainty,
                "variance": variance,
            }
            outputs.update({key: value for key, value in condition_outputs.items() if key != "raw_output"})
            return outputs

        outputs = {
            "raw_output": raw_output,
            "logits": raw_output,
            "probs": torch.sigmoid(raw_output),
        }
        outputs.update({key: value for key, value in condition_outputs.items() if key != "raw_output"})
        return outputs

    def forward(
        self,
        radar_past: torch.Tensor,
        lightning_past: torch.Tensor,
        future_condition: torch.Tensor | None = None,
        time_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.predict_distribution(
            radar_past,
            lightning_past,
            future_condition=future_condition,
            time_emb=time_emb,
        )["logits"]
