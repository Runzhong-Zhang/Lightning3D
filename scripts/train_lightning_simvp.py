from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lightning_nowcast.data import build_dataset
from lightning_nowcast.future_condition import build_future_condition_source, maybe_wrap_dataset_with_future_condition
from lightning_nowcast.losses import build_loss
from lightning_nowcast.metrics import finalize_threshold_metrics
from lightning_nowcast.models.simvp_lightning import LightningSimVPSystem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SimVP lightning nowcast variants.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a JSON config file.")
    return parser.parse_args()


def setup_ddp() -> tuple[int, int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        dist.barrier(device_ids=[local_rank])
        return rank, world_size, local_rank, torch.device(f"cuda:{local_rank}")
    return 0, 1, 0, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def reduce_sum_count_to_mean(local_sum: float, local_count: int, device: torch.device) -> float:
    sum_tensor = torch.tensor(float(local_sum), device=device)
    count_tensor = torch.tensor(float(local_count), device=device)
    if dist.is_initialized():
        dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
    return float((sum_tensor / count_tensor.clamp_min(1.0)).item())


def reduce_max_value(local_value: float, device: torch.device) -> float:
    value_tensor = torch.tensor(float(local_value), device=device)
    if dist.is_initialized():
        dist.all_reduce(value_tensor, op=dist.ReduceOp.MAX)
    return float(value_tensor.item())


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    distributed: bool = False,
    shard_across_ranks: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    if shard_across_ranks:
        if shuffle:
            raise ValueError("shard_across_ranks requires shuffle=False.")
        indices = list(range(int(rank), len(dataset), int(world_size)))
        dataset = Subset(dataset, indices)
        sampler = None
    else:
        sampler = DistributedSampler(dataset, shuffle=shuffle) if distributed else None
    dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle and sampler is None,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = 4
    return DataLoader(
        dataset,
        **dataloader_kwargs,
    )


def move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device=device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def prepare_future_condition(batch: dict, future_condition_source) -> torch.Tensor | None:
    if isinstance(batch.get("future_condition"), torch.Tensor):
        return batch["future_condition"]
    if future_condition_source is None:
        return None
    return future_condition_source.build_future_condition(
        batch,
        out_seq_len=batch["lightning_future"].shape[1],
    )


def set_future_condition_source_mode(future_condition_source, training: bool) -> None:
    if future_condition_source is None:
        return
    if training:
        future_condition_source.train()
    else:
        future_condition_source.eval()


def get_future_condition_trainable_parameters(future_condition_source) -> list[torch.nn.Parameter]:
    if future_condition_source is None:
        return []
    if not getattr(future_condition_source, "has_trainable_parameters", False):
        return []
    return list(future_condition_source.trainable_parameters())


def get_future_condition_source_state_dict(future_condition_source) -> dict[str, torch.Tensor] | None:
    if future_condition_source is None:
        return None
    predictor = getattr(future_condition_source, "predictor", None)
    if predictor is None or not hasattr(predictor, "model"):
        return None
    if not getattr(future_condition_source, "has_trainable_parameters", False):
        return None
    return predictor.model.state_dict()


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key.replace("module.", ""): value for key, value in state_dict.items()}


def _filter_compatible_state_dict(
    current_state_dict: dict[str, torch.Tensor],
    checkpoint_state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str], list[str]]:
    filtered: dict[str, torch.Tensor] = {}
    loaded_keys: list[str] = []
    skipped_keys: list[str] = []
    for key, value in checkpoint_state_dict.items():
        current_value = current_state_dict.get(key)
        if current_value is None or tuple(current_value.shape) != tuple(value.shape):
            skipped_keys.append(key)
            continue
        filtered[key] = value
        loaded_keys.append(key)
    return filtered, loaded_keys, skipped_keys


def maybe_initialize_from_checkpoint(
    model_for_eval: LightningSimVPSystem,
    future_condition_source,
    config: dict,
    device: torch.device,
    is_main_process: bool,
) -> None:
    training_config = config.get("training", {})
    init_path = training_config.get("init_from_checkpoint")
    if not init_path:
        return

    checkpoint_path = Path(init_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    checkpoint_model_state = _strip_module_prefix(checkpoint["model_state_dict"])
    filtered_model_state, loaded_model_keys, skipped_model_keys = _filter_compatible_state_dict(
        model_for_eval.state_dict(),
        checkpoint_model_state,
    )
    model_for_eval.load_state_dict(filtered_model_state, strict=False)

    loaded_source_keys: list[str] = []
    skipped_source_keys: list[str] = []
    checkpoint_source_state = checkpoint.get("future_condition_source_state_dict")
    if checkpoint_source_state is not None and future_condition_source is not None:
        predictor = getattr(future_condition_source, "predictor", None)
        if predictor is not None and hasattr(predictor, "model"):
            filtered_source_state, loaded_source_keys, skipped_source_keys = _filter_compatible_state_dict(
                predictor.model.state_dict(),
                _strip_module_prefix(checkpoint_source_state),
            )
            predictor.model.load_state_dict(filtered_source_state, strict=False)

    if is_main_process:
        print(
            json.dumps(
                {
                    "event": "init_from_checkpoint",
                    "path": str(checkpoint_path),
                    "loaded_model_keys": len(loaded_model_keys),
                    "skipped_model_keys": len(skipped_model_keys),
                    "loaded_future_condition_keys": len(loaded_source_keys),
                    "skipped_future_condition_keys": len(skipped_source_keys),
                },
                sort_keys=True,
            ),
            flush=True,
        )


def _flatten_metric_counts(metric_counts: dict[float, dict[str, float]], thresholds: list[float]) -> list[float]:
    flattened: list[float] = []
    for threshold in thresholds:
        counts = metric_counts[float(threshold)]
        flattened.extend([float(counts["tp"]), float(counts["fp"]), float(counts["fn"]), float(counts["tn"])] )
    return flattened


def _assign_flat_metric_counts(metric_counts: dict[float, dict[str, float]], thresholds: list[float], values: list[float]) -> None:
    offset = 0
    for threshold in thresholds:
        counts = metric_counts[float(threshold)]
        counts["tp"] = float(values[offset])
        counts["fp"] = float(values[offset + 1])
        counts["fn"] = float(values[offset + 2])
        counts["tn"] = float(values[offset + 3])
        offset += 4


def sync_metric_accumulator(accumulator: StreamingForecastMetricAccumulator, device: torch.device) -> None:
    if not dist.is_initialized():
        return

    thresholds = accumulator.thresholds
    packed = [
        accumulator.brier_sum,
        accumulator.truth_sum,
        float(accumulator.pixel_count),
    ]
    for metric_counts in (
        accumulator.frame_counts,
        accumulator.pooled_frame_counts,
        accumulator.hourly_counts,
        accumulator.pooled_hourly_counts,
        accumulator.default_counts,
        accumulator.pooled_default_counts,
        accumulator.hourly_default_counts,
        accumulator.pooled_hourly_default_counts,
    ):
        packed.extend(_flatten_metric_counts(metric_counts, list(metric_counts.keys())))

    packed_tensor = torch.tensor(packed, dtype=torch.float64, device=device)
    dist.all_reduce(packed_tensor, op=dist.ReduceOp.SUM)
    reduced = packed_tensor.cpu().tolist()

    accumulator.brier_sum = float(reduced[0])
    accumulator.truth_sum = float(reduced[1])
    accumulator.pixel_count = int(round(reduced[2]))

    offset = 3
    for metric_counts in (
        accumulator.frame_counts,
        accumulator.pooled_frame_counts,
        accumulator.hourly_counts,
        accumulator.pooled_hourly_counts,
        accumulator.default_counts,
        accumulator.pooled_default_counts,
        accumulator.hourly_default_counts,
        accumulator.pooled_hourly_default_counts,
    ):
        local_thresholds = list(metric_counts.keys())
        next_offset = offset + 4 * len(local_thresholds)
        _assign_flat_metric_counts(metric_counts, local_thresholds, reduced[offset:next_offset])
        offset = next_offset


def _pool_spatial_binary(arr: torch.Tensor, pool_size: int) -> torch.Tensor:
    if pool_size <= 1:
        return arr
    *prefix, height, width = arr.shape
    flat = arr.reshape(-1, 1, height, width).to(dtype=torch.float32)
    pooled = F.max_pool2d(flat, kernel_size=pool_size, stride=pool_size, ceil_mode=True)
    return (pooled > 0.5).reshape(*prefix, pooled.shape[-2], pooled.shape[-1])


def _count_binary_predictions(scores: torch.Tensor, truth: torch.Tensor, thresholds: torch.Tensor, pool_size: int = 1) -> torch.Tensor:
    thresholds_view = thresholds.view(-1, *([1] * scores.ndim))
    pred = scores.unsqueeze(0) >= thresholds_view
    true = truth.unsqueeze(0)
    pred = _pool_spatial_binary(pred, pool_size)
    true = _pool_spatial_binary(true, pool_size)
    sum_dims = tuple(range(1, pred.ndim))
    tp = torch.logical_and(pred, true).sum(dim=sum_dims, dtype=torch.float64)
    fp = torch.logical_and(pred, torch.logical_not(true)).sum(dim=sum_dims, dtype=torch.float64)
    fn = torch.logical_and(torch.logical_not(pred), true).sum(dim=sum_dims, dtype=torch.float64)
    tn = torch.logical_and(torch.logical_not(pred), torch.logical_not(true)).sum(dim=sum_dims, dtype=torch.float64)
    return torch.stack([tp, fp, fn, tn], dim=1)


class DeviceValidationMetricAccumulator:
    def __init__(self, thresholds: list[float], default_threshold: float, pooled_pool_size: int, device: torch.device):
        self.thresholds = [float(value) for value in thresholds]
        self.default_threshold = float(default_threshold)
        self.pooled_pool_size = int(pooled_pool_size)
        self.device = device

        self.threshold_tensor = torch.tensor(self.thresholds, dtype=torch.float32, device=device)
        self.default_threshold_tensor = torch.tensor([self.default_threshold], dtype=torch.float32, device=device)

        self.brier_sum = torch.zeros((), dtype=torch.float64, device=device)
        self.truth_sum = torch.zeros((), dtype=torch.float64, device=device)
        self.pixel_count = torch.zeros((), dtype=torch.float64, device=device)

        shape = (len(self.thresholds), 4)
        self.frame_counts = torch.zeros(shape, dtype=torch.float64, device=device)
        self.pooled_frame_counts = torch.zeros(shape, dtype=torch.float64, device=device)
        self.hourly_counts = torch.zeros(shape, dtype=torch.float64, device=device)
        self.pooled_hourly_counts = torch.zeros(shape, dtype=torch.float64, device=device)

        self.default_counts = torch.zeros((1, 4), dtype=torch.float64, device=device)
        self.pooled_default_counts = torch.zeros((1, 4), dtype=torch.float64, device=device)
        self.hourly_default_counts = torch.zeros((1, 4), dtype=torch.float64, device=device)
        self.pooled_hourly_default_counts = torch.zeros((1, 4), dtype=torch.float64, device=device)

    def update(self, probs: torch.Tensor, target: torch.Tensor) -> None:
        scores = probs.squeeze(2).to(dtype=torch.float32)
        truth = target.squeeze(2) >= 0.5
        probs_clamped = scores.clamp_(0.0, 1.0)
        truth_float = truth.to(dtype=torch.float32)

        self.brier_sum += torch.square(probs_clamped.to(dtype=torch.float64) - truth_float.to(dtype=torch.float64)).sum()
        self.truth_sum += truth.sum(dtype=torch.float64)
        self.pixel_count += torch.tensor(float(truth.numel()), dtype=torch.float64, device=self.device)

        self.frame_counts += _count_binary_predictions(scores, truth, self.threshold_tensor, pool_size=1)
        self.pooled_frame_counts += _count_binary_predictions(
            scores,
            truth,
            self.threshold_tensor,
            pool_size=self.pooled_pool_size,
        )
        self.default_counts += _count_binary_predictions(scores, truth, self.default_threshold_tensor, pool_size=1)
        self.pooled_default_counts += _count_binary_predictions(
            scores,
            truth,
            self.default_threshold_tensor,
            pool_size=self.pooled_pool_size,
        )

        hourly_scores = scores.max(dim=1).values
        hourly_truth = truth.any(dim=1)
        self.hourly_counts += _count_binary_predictions(hourly_scores, hourly_truth, self.threshold_tensor, pool_size=1)
        self.pooled_hourly_counts += _count_binary_predictions(
            hourly_scores,
            hourly_truth,
            self.threshold_tensor,
            pool_size=self.pooled_pool_size,
        )
        self.hourly_default_counts += _count_binary_predictions(
            hourly_scores,
            hourly_truth,
            self.default_threshold_tensor,
            pool_size=1,
        )
        self.pooled_hourly_default_counts += _count_binary_predictions(
            hourly_scores,
            hourly_truth,
            self.default_threshold_tensor,
            pool_size=self.pooled_pool_size,
        )

    def _all_reduce_(self) -> None:
        if not dist.is_initialized():
            return
        for tensor in (
            self.brier_sum,
            self.truth_sum,
            self.pixel_count,
            self.frame_counts,
            self.pooled_frame_counts,
            self.hourly_counts,
            self.pooled_hourly_counts,
            self.default_counts,
            self.pooled_default_counts,
            self.hourly_default_counts,
            self.pooled_hourly_default_counts,
        ):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    def _counts_to_dict(self, counts_tensor: torch.Tensor, thresholds: list[float]) -> dict[float, dict[str, float]]:
        counts_np = counts_tensor.detach().cpu().tolist()
        return {
            float(threshold): {
                "tp": float(values[0]),
                "fp": float(values[1]),
                "fn": float(values[2]),
                "tn": float(values[3]),
            }
            for threshold, values in zip(thresholds, counts_np)
        }

    def finalize(self) -> dict[str, float]:
        self._all_reduce_()
        overall, _ = finalize_threshold_metrics(self._counts_to_dict(self.frame_counts, self.thresholds))
        pooled_overall, _ = finalize_threshold_metrics(self._counts_to_dict(self.pooled_frame_counts, self.thresholds))
        hourly_overall, _ = finalize_threshold_metrics(self._counts_to_dict(self.hourly_counts, self.thresholds))
        pooled_hourly_overall, _ = finalize_threshold_metrics(self._counts_to_dict(self.pooled_hourly_counts, self.thresholds))

        default_metrics, _ = finalize_threshold_metrics(self._counts_to_dict(self.default_counts, [self.default_threshold]))
        pooled_default_metrics, _ = finalize_threshold_metrics(
            self._counts_to_dict(self.pooled_default_counts, [self.default_threshold])
        )
        hourly_default_metrics, _ = finalize_threshold_metrics(
            self._counts_to_dict(self.hourly_default_counts, [self.default_threshold])
        )
        pooled_hourly_default_metrics, _ = finalize_threshold_metrics(
            self._counts_to_dict(self.pooled_hourly_default_counts, [self.default_threshold])
        )

        eps = 1.0e-8
        pixel_count = max(float(self.pixel_count.item()), 1.0)
        brier = float(self.brier_sum.item() / pixel_count)
        clim = float(self.truth_sum.item() / pixel_count)
        bss = float(1.0 - brier / (clim * (1.0 - clim) + eps))

        return {
            "brier": brier,
            "bss": bss,
            "default_threshold": float(self.default_threshold),
            "pooled_pool_size": int(self.pooled_pool_size),
            "best_threshold": float(overall["threshold"]),
            "best_csi": float(overall["csi"]),
            "best_pod": float(overall["pod"]),
            "best_far": float(overall["far"]),
            "pooled_best_threshold": float(pooled_overall["threshold"]),
            "pooled_best_csi": float(pooled_overall["csi"]),
            "pooled_best_pod": float(pooled_overall["pod"]),
            "pooled_best_far": float(pooled_overall["far"]),
            "hourly_best_threshold": float(hourly_overall["threshold"]),
            "hourly_best_csi": float(hourly_overall["csi"]),
            "hourly_best_pod": float(hourly_overall["pod"]),
            "hourly_best_far": float(hourly_overall["far"]),
            "pooled_hourly_best_threshold": float(pooled_hourly_overall["threshold"]),
            "pooled_hourly_best_csi": float(pooled_hourly_overall["csi"]),
            "pooled_hourly_best_pod": float(pooled_hourly_overall["pod"]),
            "pooled_hourly_best_far": float(pooled_hourly_overall["far"]),
            "default_csi": float(default_metrics["csi"]),
            "default_pod": float(default_metrics["pod"]),
            "default_far": float(default_metrics["far"]),
            "pooled_default_csi": float(pooled_default_metrics["csi"]),
            "pooled_default_pod": float(pooled_default_metrics["pod"]),
            "pooled_default_far": float(pooled_default_metrics["far"]),
            "hourly_default_csi": float(hourly_default_metrics["csi"]),
            "hourly_default_pod": float(hourly_default_metrics["pod"]),
            "hourly_default_far": float(hourly_default_metrics["far"]),
            "pooled_hourly_default_csi": float(pooled_hourly_default_metrics["csi"]),
            "pooled_hourly_default_pod": float(pooled_hourly_default_metrics["pod"]),
            "pooled_hourly_default_far": float(pooled_hourly_default_metrics["far"]),
        }


def evaluate(
    model: LightningSimVPSystem,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn,
    thresholds: list[float],
    future_condition_source,
    default_threshold: float,
    pooled_pool_size: int,
) -> dict[str, float]:
    model.eval()
    set_future_condition_source_mode(future_condition_source, training=False)
    total_loss = 0.0
    total_examples = 0
    metric_accumulator = DeviceValidationMetricAccumulator(
        thresholds=thresholds,
        default_threshold=default_threshold,
        pooled_pool_size=pooled_pool_size,
        device=device,
    )

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch(batch, device)
            future_condition = prepare_future_condition(batch, future_condition_source)
            logits = model(
                batch["radar_past"],
                batch["lightning_past"],
                future_condition=future_condition,
                time_emb=batch.get("time_emb"),
            )
            probs = torch.sigmoid(logits)
            target = batch["lightning_future"].permute(0, 1, 4, 2, 3).contiguous()
            loss = loss_fn(logits, target)
            batch_size = int(target.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            metric_accumulator.update(probs, target)

    total_loss = reduce_sum_count_to_mean(total_loss, total_examples, device)
    summary = metric_accumulator.finalize()
    return {
        "loss": total_loss,
        **summary,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    rank, world_size, local_rank, device = setup_ddp()
    is_main_process = rank == 0
    set_seed(int(config["seed"]) + rank)

    output_dir = ROOT / config["output_dir"] / config["experiment_name"]
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier(device_ids=[local_rank])

    train_dataset = build_dataset("train", config["data"])
    val_dataset = build_dataset("val", config["data"])
    train_dataset = maybe_wrap_dataset_with_future_condition(train_dataset, config, "train")
    val_dataset = maybe_wrap_dataset_with_future_condition(val_dataset, config, "val")

    train_loader = build_dataloader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["training"]["num_workers"]),
        shuffle=True,
        distributed=world_size > 1,
        rank=rank,
        world_size=world_size,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        num_workers=int(config["evaluation"]["num_workers"]),
        shuffle=False,
        distributed=False,
        shard_across_ranks=world_size > 1,
        rank=rank,
        world_size=world_size,
    )

    model = LightningSimVPSystem(**config["model"]).to(device)
    model_for_eval = model
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        model_for_eval = model.module

    train_future_condition_source = None
    val_future_condition_source = None
    if config["radar_condition"]["enabled"]:
        source_name = str(config["radar_condition"].get("source", "online")).lower()
        if source_name == "online":
            train_future_condition_source = build_future_condition_source(config, "train", device)
            val_future_condition_source = train_future_condition_source

    maybe_initialize_from_checkpoint(
        model_for_eval=model_for_eval,
        future_condition_source=train_future_condition_source,
        config=config,
        device=device,
        is_main_process=is_main_process,
    )

    extra_optimizer_parameters = get_future_condition_trainable_parameters(train_future_condition_source)
    if extra_optimizer_parameters and world_size > 1:
        raise ValueError(
            "Trainable online radar-condition modules are only supported in single-process training. "
            "Launch with plain python on one GPU when using radar_condition.predictor.trainable_modules."
        )

    optimizer_parameters = list(model.parameters())
    optimizer_parameters.extend(extra_optimizer_parameters)

    optimizer = AdamW(
        optimizer_parameters,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=int(config["training"]["epochs"]),
        eta_min=float(config["training"]["learning_rate"]) * float(config["training"]["min_lr_ratio"]),
    )
    loss_fn = build_loss(
        config["training"]["loss_name"],
        float(config["training"]["focal_alpha"]),
        float(config["training"]["focal_gamma"]),
        config["training"].get("pos_weight"),
    )

    thresholds = [float(value) for value in config["evaluation"]["thresholds"]]
    default_threshold = float(config["evaluation"].get("default_threshold", 0.5))
    pooled_pool_size = int(config["evaluation"].get("pooled_pool_size", 2))
    best_score = -math.inf
    history = []

    try:
        for epoch in range(1, int(config["training"]["epochs"]) + 1):
            sync_device(device)
            epoch_start_time = time.perf_counter()
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
            model.train()
            set_future_condition_source_mode(train_future_condition_source, training=True)
            train_loss_sum = 0.0
            train_example_count = 0
            num_batches = len(train_loader)

            for batch_idx, batch in enumerate(train_loader, start=1):
                batch = move_batch(batch, device)

                future_condition = prepare_future_condition(
                    batch,
                    train_future_condition_source
                )

                target = batch["lightning_future"].permute(
                    0, 1, 4, 2, 3
                ).contiguous()

                optimizer.zero_grad(set_to_none=True)

                logits = model(
                    batch["radar_past"],
                    batch["lightning_past"],
                    future_condition=future_condition,
                    time_emb=batch.get("time_emb"),
                )

                loss = loss_fn(logits, target)

                loss.backward()

                if config["training"].get("gradient_clip_norm") is not None:
                    torch.nn.utils.clip_grad_norm_(
                        optimizer_parameters,
                        float(config["training"]["gradient_clip_norm"])
                    )

                optimizer.step()

                batch_size = int(target.shape[0])
                train_loss_sum += float(loss.item()) * batch_size
                train_example_count += batch_size

                if batch_idx % 10 == 0 or batch_idx == 1:
                    print(
                        f"Epoch {epoch} | "
                        f"Batch {batch_idx}/{num_batches} | "
                        f"Loss: {loss.item():.6f}",
                        flush=True,
                    )

            '''for batch in train_loader:
                batch = move_batch(batch, device)
                future_condition = prepare_future_condition(batch, train_future_condition_source)
                target = batch["lightning_future"].permute(0, 1, 4, 2, 3).contiguous()

                optimizer.zero_grad(set_to_none=True)
                logits = model(
                    batch["radar_past"],
                    batch["lightning_past"],
                    future_condition=future_condition,
                    time_emb=batch.get("time_emb"),
                )
                loss = loss_fn(logits, target)
                loss.backward()
                if config["training"].get("gradient_clip_norm") is not None:
                    torch.nn.utils.clip_grad_norm_(optimizer_parameters, float(config["training"]["gradient_clip_norm"]))
                optimizer.step()

                batch_size = int(target.shape[0])
                train_loss_sum += float(loss.item()) * batch_size
                train_example_count += batch_size'''
                

            # sync_device(device)
            # train_elapsed = time.perf_counter() - epoch_start_time
            # scheduler.step()
            # train_loss = reduce_sum_count_to_mean(train_loss_sum, train_example_count, device)
            # if dist.is_initialized():
            #     dist.barrier(device_ids=[local_rank])

            # sync_device(device)
            # val_start_time = time.perf_counter()
            # val_metrics = evaluate(
            #     model_for_eval,
            #     val_loader,
            #     device,
            #     loss_fn,
            #     thresholds,
            #     val_future_condition_source,
            #     default_threshold,
            #     pooled_pool_size,
            # )
            # sync_device(device)
            # val_elapsed = time.perf_counter() - val_start_time
            # epoch_elapsed = time.perf_counter() - epoch_start_time

            # train_elapsed = reduce_max_value(train_elapsed, device)
            # val_elapsed = reduce_max_value(val_elapsed, device)
            # epoch_elapsed = reduce_max_value(epoch_elapsed, device)
            # val_fraction = val_elapsed / max(epoch_elapsed, 1.0e-8)
            # sync_device(device)

            # epoch_elapsed = time.perf_counter() - epoch_start_time

            # train_elapsed = reduce_max_value(train_elapsed, device)
            # epoch_elapsed = reduce_max_value(epoch_elapsed, device)

            # #val_metrics = {}
            # #val_elapsed = 0.0
            # #val_fraction = 0.0

            # if is_main_process:
            #     epoch_metrics = {
            #         "epoch": epoch,
            #         "train_loss": train_loss,
            #         "train_seconds": train_elapsed,
            #         "val_seconds": val_elapsed,
            #         "epoch_seconds": epoch_elapsed,
            #         "val_fraction": val_fraction,
            #         **val_metrics,
            #     }
            #     history.append(epoch_metrics)
            #     print(
            #         f"TIMING epoch={epoch} train_s={train_elapsed:.2f} "
            #         f"val_s={val_elapsed:.2f} total_s={epoch_elapsed:.2f} "
            #         f"val_frac={val_fraction:.3f}"
            #     )
            #     print(json.dumps(epoch_metrics, sort_keys=True))

            #     if val_metrics["best_csi"] > best_score:
            #         best_score = val_metrics["best_csi"]
            #         checkpoint = {
            #             "epoch": epoch,
            #             "config": config,
            #             "model_state_dict": model_for_eval.state_dict(),
            #             "future_condition_source_state_dict": get_future_condition_source_state_dict(train_future_condition_source),
            #             "optimizer_state_dict": optimizer.state_dict(),
            #             "scheduler_state_dict": scheduler.state_dict(),
            #             "val_metrics": val_metrics,
            #         }
            #         torch.save(checkpoint, output_dir / "best.pt")

            #     torch.save(
            #         {
            #             "epoch": epoch,
            #             "config": config,
            #             "model_state_dict": model_for_eval.state_dict(),
            #             "future_condition_source_state_dict": get_future_condition_source_state_dict(train_future_condition_source),
            #             "optimizer_state_dict": optimizer.state_dict(),
            #             "scheduler_state_dict": scheduler.state_dict(),
            #             "history": history,
            #         },
            #         output_dir / "latest.pt",
            #     )

            # if dist.is_initialized():
            #     dist.barrier(device_ids=[local_rank])
            sync_device(device)
            train_elapsed = time.perf_counter() - epoch_start_time
            scheduler.step()
            train_loss = reduce_sum_count_to_mean(train_loss_sum, train_example_count, device)
            if dist.is_initialized():
                dist.barrier(device_ids=[local_rank])

            sync_device(device)
            val_start_time = time.perf_counter()
            val_metrics = evaluate(
                model_for_eval,
                val_loader,
                device,
                loss_fn,
                thresholds,
                val_future_condition_source,
                default_threshold,
                pooled_pool_size,
            )
            sync_device(device)
            val_elapsed = time.perf_counter() - val_start_time
            epoch_elapsed = time.perf_counter() - epoch_start_time

            train_elapsed = reduce_max_value(train_elapsed, device)
            val_elapsed = reduce_max_value(val_elapsed, device)
            epoch_elapsed = reduce_max_value(epoch_elapsed, device)
            val_fraction = val_elapsed / max(epoch_elapsed, 1.0e-8)

            if is_main_process:
                epoch_metrics = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_seconds": train_elapsed,
                    "val_seconds": val_elapsed,
                    "epoch_seconds": epoch_elapsed,
                    "val_fraction": val_fraction,
                    **val_metrics,
                }
                history.append(epoch_metrics)
                print(
                    f"TIMING epoch={epoch} train_s={train_elapsed:.2f} "
                    f"val_s={val_elapsed:.2f} total_s={epoch_elapsed:.2f} "
                    f"val_frac={val_fraction:.3f}"
                )
                print(json.dumps(epoch_metrics, sort_keys=True))

                if val_metrics["best_csi"] > best_score:
                    best_score = val_metrics["best_csi"]
                    checkpoint = {
                        "epoch": epoch,
                        "config": config,
                        "model_state_dict": model_for_eval.state_dict(),
                        "future_condition_source_state_dict": get_future_condition_source_state_dict(train_future_condition_source),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "val_metrics": val_metrics,
                    }
                    torch.save(checkpoint, output_dir / "best.pt")

                torch.save(
                    {
                        "epoch": epoch,
                        "config": config,
                        "model_state_dict": model_for_eval.state_dict(),
                        "future_condition_source_state_dict": get_future_condition_source_state_dict(train_future_condition_source),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "history": history,
                    },
                    output_dir / "latest.pt",
                )

            if dist.is_initialized():
                dist.barrier(device_ids=[local_rank])

        if is_main_process:
            with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
                json.dump(history, handle, indent=2)
    finally:
        for dataset in (train_dataset, val_dataset):
            if hasattr(dataset, "close"):
                dataset.close()
        if val_future_condition_source is not None and val_future_condition_source is not train_future_condition_source:
            val_future_condition_source.close()
        if train_future_condition_source is not None:
            train_future_condition_source.close()
        cleanup_ddp()


if __name__ == "__main__":
    main()
