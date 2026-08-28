from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lightning_nowcast.data import build_dataset, infer_lightning_spatial_mask, spatial_mask_to_bbox
from lightning_nowcast.future_condition import build_future_condition_source, maybe_wrap_dataset_with_future_condition
from lightning_nowcast.losses import build_loss
from lightning_nowcast.metrics import build_per_lead_metrics_from_counts, finalize_threshold_metrics
from lightning_nowcast.models.simvp_lightning import LightningSimVPSystem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained SimVP lightning nowcast model.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a saved training checkpoint.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to evaluate.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Explicit single-process device, for example cuda:0 or cpu. Use plain python instead of torchrun when set.",
    )
    parser.add_argument(
        "--export-branch-attention",
        action="store_true",
        help="Export interval-attention branch maps and branch-conditioned radar values to HDF5.",
    )
    parser.add_argument(
        "--branch-attention-output",
        type=Path,
        default=None,
        help="Optional output path for branch-attention HDF5 export. Defaults next to the checkpoint.",
    )
    parser.add_argument(
        "--export-predictions",
        action="store_true",
        help="Export predicted lightning probability and optional uncertainty maps to HDF5.",
    )
    parser.add_argument(
        "--prediction-output",
        type=Path,
        default=None,
        help="Optional output path for prediction HDF5 export. Defaults next to the checkpoint.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print a rank-0 evaluation progress update every N local batches. Use 0 to disable.",
    )
    parser.add_argument(
        "--prepare-spatial-mask-only",
        action="store_true",
        help="Infer and cache the spatial mask for this split, then exit without running model evaluation.",
    )
    return parser.parse_args()


def resolve_single_process_device(device_arg: str | None) -> torch.device:
    if device_arg is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested for evaluation, but CUDA is not available.")
        torch.cuda.set_device(device)
    return device


def setup_ddp(args: argparse.Namespace) -> tuple[int, int, int, torch.device]:
    ddp_env_keys = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    present_ddp_env = [key for key in ddp_env_keys if key in os.environ]
    if args.device is not None:
        if present_ddp_env:
            raise RuntimeError(
                "Single-process device mode was requested with --device, but distributed env vars are present. "
                "Launch with plain python instead of torchrun when forcing one GPU."
            )
        return 0, 1, 0, resolve_single_process_device(args.device)
    if present_ddp_env and len(present_ddp_env) != len(ddp_env_keys):
        missing_ddp_env = [key for key in ddp_env_keys if key not in os.environ]
        raise RuntimeError(
            "Incomplete distributed launch environment detected. "
            f"Present: {present_ddp_env}. Missing: {missing_ddp_env}. "
            "Use torchrun for multi-GPU evaluation or clear partial DDP env vars for single-GPU runs."
        )
    if len(present_ddp_env) == len(ddp_env_keys):
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        dist.barrier(device_ids=[local_rank])
        return rank, world_size, local_rank, torch.device(f"cuda:{local_rank}")
    return 0, 1, 0, resolve_single_process_device(None)


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def shard_dataset_across_ranks(dataset: Dataset, rank: int, world_size: int) -> Dataset:
    if world_size <= 1:
        return dataset
    indices = list(range(int(rank), len(dataset), int(world_size)))
    return Subset(dataset, indices)


def build_dataloader(dataset: Dataset, batch_size: int, num_workers: int) -> DataLoader:
    dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **dataloader_kwargs)


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


def load_future_condition_source_state(future_condition_source, checkpoint: dict) -> None:
    state_dict = checkpoint.get("future_condition_source_state_dict")
    if state_dict is None:
        return
    if future_condition_source is None:
        raise ValueError(
            "Checkpoint includes future_condition_source_state_dict, but evaluation did not build an online future-condition source."
        )
    predictor = getattr(future_condition_source, "predictor", None)
    if predictor is None or not hasattr(predictor, "model"):
        raise ValueError("Future-condition source does not expose a predictor model for checkpoint loading.")
    predictor.model.load_state_dict(state_dict, strict=True)


def build_spatial_mask_cache_path(data_config: dict, target_hw: tuple[int, int]) -> Path:
    cache_key = json.dumps(
        {
            "target_hw": [int(target_hw[0]), int(target_hw[1])],
            "data": data_config,
        },
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:16]
    return ROOT / "outputs" / "cache" / f"spatial_mask_shared_{digest}.npz"


def build_rectangular_spatial_mask(target_hw: tuple[int, int], bbox: dict[str, int]) -> np.ndarray:
    row_min = int(bbox["row_min"])
    row_max = int(bbox["row_max"])
    col_min = int(bbox["col_min"])
    col_max = int(bbox["col_max"])
    height, width = int(target_hw[0]), int(target_hw[1])
    if not (0 <= row_min <= row_max < height and 0 <= col_min <= col_max < width):
        raise ValueError(f"Spatial mask bbox {bbox} is outside target grid {target_hw}")
    spatial_mask = np.zeros((height, width), dtype=bool)
    spatial_mask[row_min : row_max + 1, col_min : col_max + 1] = True
    return spatial_mask


def load_spatial_mask_from_metrics_json(metrics_path: Path, target_hw: tuple[int, int]) -> np.ndarray | None:
    if not metrics_path.is_file():
        return None
    with metrics_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    bbox = summary.get("spatial_mask_bbox")
    if not isinstance(bbox, dict):
        return None
    expected_pixel_count = (
        (int(bbox["row_max"]) - int(bbox["row_min"]) + 1)
        * (int(bbox["col_max"]) - int(bbox["col_min"]) + 1)
    )
    pixel_count = summary.get("spatial_mask_pixel_count")
    if pixel_count is not None and int(pixel_count) != expected_pixel_count:
        return None
    return build_rectangular_spatial_mask(target_hw, bbox)


def resolve_precomputed_spatial_mask(
    config: dict,
    checkpoint_path: Path,
    split: str,
    target_hw: tuple[int, int],
) -> tuple[np.ndarray | None, str | None]:
    for section_name in ("data", "evaluation"):
        section = config.get(section_name, {})
        bbox = section.get("spatial_mask_bbox") if isinstance(section, dict) else None
        if isinstance(bbox, dict):
            return build_rectangular_spatial_mask(target_hw, bbox), f"config:{section_name}"

    candidate_paths = [
        checkpoint_path.parent / f"metrics_{split}.json",
        checkpoint_path.parent / "metrics_val.json",
        checkpoint_path.parent / "metrics_test.json",
        checkpoint_path.parent / "metrics_train.json",
    ]
    seen_paths: set[Path] = set()
    for candidate_path in candidate_paths:
        if candidate_path in seen_paths:
            continue
        seen_paths.add(candidate_path)
        spatial_mask = load_spatial_mask_from_metrics_json(candidate_path, target_hw)
        if spatial_mask is not None:
            return spatial_mask, f"metrics:{candidate_path.name}"
    return None, None


def load_spatial_mask_cache(cache_path: Path, target_hw: tuple[int, int]) -> np.ndarray:
    with np.load(cache_path) as handle:
        spatial_mask = np.asarray(handle["spatial_mask"], dtype=bool)
    if spatial_mask.shape != tuple(target_hw):
        raise ValueError(f"Expected cached spatial_mask shape {target_hw}, got {spatial_mask.shape}")
    return spatial_mask


def save_spatial_mask_cache(cache_path: Path, spatial_mask: np.ndarray) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_name(f"{cache_path.stem}.tmp-{os.getpid()}{cache_path.suffix}")
    try:
        np.savez_compressed(temp_path, spatial_mask=np.asarray(spatial_mask, dtype=np.uint8))
        os.replace(temp_path, cache_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def infer_and_sync_spatial_mask(
    dataset: Dataset,
    target_hw: tuple[int, int],
    rank: int,
    device: torch.device,
    precomputed_mask: np.ndarray | None = None,
    precomputed_mask_source: str | None = None,
    cache_path: Path | None = None,
) -> tuple[np.ndarray, dict[str, int] | None, str]:
    if not dist.is_initialized() or rank == 0:
        if precomputed_mask is not None:
            spatial_mask = np.asarray(precomputed_mask, dtype=bool)
            mask_source = precomputed_mask_source or "precomputed"
            if cache_path is not None and not cache_path.is_file():
                save_spatial_mask_cache(cache_path, spatial_mask)
        elif cache_path is not None and cache_path.is_file():
            spatial_mask = load_spatial_mask_cache(cache_path, target_hw)
            mask_source = "cache"
        else:
            spatial_mask = infer_lightning_spatial_mask(dataset)
            mask_source = "inferred"
            if cache_path is not None:
                save_spatial_mask_cache(cache_path, spatial_mask)
        mask_tensor = torch.from_numpy(spatial_mask.astype(np.uint8, copy=False)).to(device=device)
    else:
        mask_source = "broadcast"
        mask_tensor = torch.empty(target_hw, dtype=torch.uint8, device=device)

    if dist.is_initialized():
        dist.broadcast(mask_tensor, src=0)

    mask_np = mask_tensor.cpu().numpy().astype(bool, copy=False)
    return mask_np, spatial_mask_to_bbox(mask_np), mask_source


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reduce_loss_sum_and_count(local_loss_sum: float, local_count: int, device: torch.device) -> float:
    loss_tensor = torch.tensor(float(local_loss_sum), dtype=torch.float64, device=device)
    count_tensor = torch.tensor(float(local_count), dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
    return float((loss_tensor / count_tensor.clamp_min(1.0)).item())


def _pool_spatial_binary(arr: torch.Tensor, pool_size: int) -> torch.Tensor:
    if pool_size <= 1:
        return arr
    *prefix, height, width = arr.shape
    flat = arr.reshape(-1, 1, height, width).to(dtype=torch.float32)
    pooled = F.max_pool2d(flat, kernel_size=pool_size, stride=pool_size, ceil_mode=True)
    return (pooled > 0.5).reshape(*prefix, pooled.shape[-2], pooled.shape[-1])


def _broadcast_spatial_mask(mask: torch.Tensor | None, array: torch.Tensor) -> torch.Tensor | None:
    if mask is None:
        return None
    return mask.view(*([1] * (array.ndim - 2)), mask.shape[0], mask.shape[1]).expand(array.shape)


def _count_binary_predictions(
    scores: torch.Tensor,
    truth: torch.Tensor,
    thresholds: torch.Tensor,
    pool_size: int = 1,
    spatial_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    thresholds_view = thresholds.view(-1, *([1] * scores.ndim))
    pred = scores.unsqueeze(0) >= thresholds_view
    true = truth.unsqueeze(0)
    valid = _broadcast_spatial_mask(spatial_mask, pred)
    if valid is not None:
        pred = torch.logical_and(pred, valid)
        true = torch.logical_and(true, valid)
    pred = _pool_spatial_binary(pred, pool_size)
    true = _pool_spatial_binary(true, pool_size)
    pooled_valid = None if valid is None else _pool_spatial_binary(valid, pool_size)

    tp = torch.logical_and(pred, true)
    fp = torch.logical_and(pred, torch.logical_not(true))
    fn = torch.logical_and(torch.logical_not(pred), true)
    tn = torch.logical_and(torch.logical_not(pred), torch.logical_not(true))
    if pooled_valid is not None:
        tp = torch.logical_and(tp, pooled_valid)
        fp = torch.logical_and(fp, pooled_valid)
        fn = torch.logical_and(fn, pooled_valid)
        tn = torch.logical_and(tn, pooled_valid)

    sum_dims = tuple(range(1, tp.ndim))
    return torch.stack(
        [
            tp.sum(dim=sum_dims, dtype=torch.float64),
            fp.sum(dim=sum_dims, dtype=torch.float64),
            fn.sum(dim=sum_dims, dtype=torch.float64),
            tn.sum(dim=sum_dims, dtype=torch.float64),
        ],
        dim=1,
    )


class DeviceEvaluationMetricAccumulator:
    def __init__(
        self,
        thresholds: list[float],
        default_threshold: float,
        pooled_pool_sizes: list[int],
        spatial_mask: np.ndarray | None,
        device: torch.device,
    ):
        self.thresholds = [float(value) for value in thresholds]
        self.default_threshold = float(default_threshold)
        self.pooled_pool_sizes = [int(value) for value in pooled_pool_sizes]
        self.primary_pool_size = int(self.pooled_pool_sizes[0])
        self.device = device

        self.threshold_tensor = torch.tensor(self.thresholds, dtype=torch.float32, device=device)
        self.default_threshold_tensor = torch.tensor([self.default_threshold], dtype=torch.float32, device=device)
        self.spatial_mask = None if spatial_mask is None else torch.from_numpy(spatial_mask.astype(bool, copy=False)).to(device=device)

        self.brier_sum = torch.zeros((), dtype=torch.float64, device=device)
        self.truth_sum = torch.zeros((), dtype=torch.float64, device=device)
        self.pixel_count = torch.zeros((), dtype=torch.float64, device=device)

        shape = (len(self.thresholds), 4)
        self.frame_counts = torch.zeros(shape, dtype=torch.float64, device=device)
        self.hourly_counts = torch.zeros(shape, dtype=torch.float64, device=device)
        self.per_lead_counts: list[torch.Tensor] | None = None
        self.pooled_frame_counts_by_size = {
            int(pool_size): torch.zeros(shape, dtype=torch.float64, device=device)
            for pool_size in self.pooled_pool_sizes
        }
        self.pooled_hourly_counts_by_size = {
            int(pool_size): torch.zeros(shape, dtype=torch.float64, device=device)
            for pool_size in self.pooled_pool_sizes
        }
        self.pooled_per_lead_counts_by_size: dict[int, list[torch.Tensor]] | None = None

        self.default_counts = torch.zeros((1, 4), dtype=torch.float64, device=device)
        self.hourly_default_counts = torch.zeros((1, 4), dtype=torch.float64, device=device)
        self.pooled_default_counts_by_size = {
            int(pool_size): torch.zeros((1, 4), dtype=torch.float64, device=device)
            for pool_size in self.pooled_pool_sizes
        }
        self.pooled_hourly_default_counts_by_size = {
            int(pool_size): torch.zeros((1, 4), dtype=torch.float64, device=device)
            for pool_size in self.pooled_pool_sizes
        }

    def update(self, probs: torch.Tensor, target: torch.Tensor) -> None:
        scores = probs.squeeze(2).to(dtype=torch.float32)
        truth = target.squeeze(2) >= 0.5
        if self.per_lead_counts is None:
            lead_time = int(scores.shape[1])
            self.per_lead_counts = [
                torch.zeros((len(self.thresholds), 4), dtype=torch.float64, device=self.device)
                for _ in range(lead_time)
            ]
            self.pooled_per_lead_counts_by_size = {
                int(pool_size): [
                    torch.zeros((len(self.thresholds), 4), dtype=torch.float64, device=self.device)
                    for _ in range(lead_time)
                ]
                for pool_size in self.pooled_pool_sizes
            }
        probs_clamped = scores.clamp(0.0, 1.0)
        truth_float = truth.to(dtype=torch.float32)
        valid = _broadcast_spatial_mask(self.spatial_mask, probs_clamped)

        brier_error = torch.square(probs_clamped.to(dtype=torch.float64) - truth_float.to(dtype=torch.float64))
        if valid is not None:
            self.brier_sum += brier_error[valid].sum()
            self.truth_sum += truth[valid].sum(dtype=torch.float64)
            self.pixel_count += valid.sum(dtype=torch.float64)
        else:
            self.brier_sum += brier_error.sum()
            self.truth_sum += truth.sum(dtype=torch.float64)
            self.pixel_count += torch.tensor(float(truth.numel()), dtype=torch.float64, device=self.device)

        self.frame_counts += _count_binary_predictions(scores, truth, self.threshold_tensor, 1, self.spatial_mask)
        self.default_counts += _count_binary_predictions(scores, truth, self.default_threshold_tensor, 1, self.spatial_mask)
        for pool_size in self.pooled_pool_sizes:
            self.pooled_frame_counts_by_size[int(pool_size)] += _count_binary_predictions(
                scores,
                truth,
                self.threshold_tensor,
                int(pool_size),
                self.spatial_mask,
            )
            self.pooled_default_counts_by_size[int(pool_size)] += _count_binary_predictions(
                scores,
                truth,
                self.default_threshold_tensor,
                int(pool_size),
                self.spatial_mask,
            )

        hourly_scores = scores.max(dim=1).values
        hourly_truth = truth.any(dim=1)
        self.hourly_counts += _count_binary_predictions(hourly_scores, hourly_truth, self.threshold_tensor, 1, self.spatial_mask)
        self.hourly_default_counts += _count_binary_predictions(
            hourly_scores,
            hourly_truth,
            self.default_threshold_tensor,
            1,
            self.spatial_mask,
        )
        for pool_size in self.pooled_pool_sizes:
            self.pooled_hourly_counts_by_size[int(pool_size)] += _count_binary_predictions(
                hourly_scores,
                hourly_truth,
                self.threshold_tensor,
                int(pool_size),
                self.spatial_mask,
            )

        assert self.per_lead_counts is not None and self.pooled_per_lead_counts_by_size is not None
        for lead_idx in range(int(scores.shape[1])):
            self.per_lead_counts[lead_idx] += _count_binary_predictions(
                scores[:, lead_idx],
                truth[:, lead_idx],
                self.threshold_tensor,
                1,
                self.spatial_mask,
            )
            for pool_size in self.pooled_pool_sizes:
                self.pooled_per_lead_counts_by_size[int(pool_size)][lead_idx] += _count_binary_predictions(
                    scores[:, lead_idx],
                    truth[:, lead_idx],
                    self.threshold_tensor,
                    int(pool_size),
                    self.spatial_mask,
                )
            self.pooled_hourly_default_counts_by_size[int(pool_size)] += _count_binary_predictions(
                hourly_scores,
                hourly_truth,
                self.default_threshold_tensor,
                int(pool_size),
                self.spatial_mask,
            )

    def _all_reduce_(self) -> None:
        if not dist.is_initialized():
            return
        for tensor in (
            self.brier_sum,
            self.truth_sum,
            self.pixel_count,
            self.frame_counts,
            self.hourly_counts,
            self.default_counts,
            self.hourly_default_counts,
        ):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        for counts_by_size in (
            self.pooled_frame_counts_by_size,
            self.pooled_hourly_counts_by_size,
            self.pooled_default_counts_by_size,
            self.pooled_hourly_default_counts_by_size,
        ):
            for tensor in counts_by_size.values():
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if self.per_lead_counts is not None:
            for tensor in self.per_lead_counts:
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if self.pooled_per_lead_counts_by_size is not None:
            for per_lead_list in self.pooled_per_lead_counts_by_size.values():
                for tensor in per_lead_list:
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    @staticmethod
    def _counts_to_dict(counts_tensor: torch.Tensor, thresholds: list[float]) -> dict[float, dict[str, float]]:
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
        overall, threshold_sweep = finalize_threshold_metrics(self._counts_to_dict(self.frame_counts, self.thresholds))
        hourly_overall, hourly_threshold_sweep = finalize_threshold_metrics(self._counts_to_dict(self.hourly_counts, self.thresholds))

        default_metrics, _ = finalize_threshold_metrics(self._counts_to_dict(self.default_counts, [self.default_threshold]))
        hourly_default_metrics, _ = finalize_threshold_metrics(
            self._counts_to_dict(self.hourly_default_counts, [self.default_threshold])
        )

        pooled_summaries_by_size: dict[str, dict[str, float | list[dict[str, float]]]] = {}
        pooled_default_metrics_by_size: dict[int, dict[str, float]] = {}
        pooled_hourly_default_metrics_by_size: dict[int, dict[str, float]] = {}
        per_lead_metrics_by_size: dict[str, list[dict[str, float]]] = {}
        assert self.per_lead_counts is not None and self.pooled_per_lead_counts_by_size is not None
        for pool_size in self.pooled_pool_sizes:
            pooled_overall, pooled_threshold_sweep = finalize_threshold_metrics(
                self._counts_to_dict(self.pooled_frame_counts_by_size[int(pool_size)], self.thresholds)
            )
            pooled_hourly_overall, pooled_hourly_threshold_sweep = finalize_threshold_metrics(
                self._counts_to_dict(self.pooled_hourly_counts_by_size[int(pool_size)], self.thresholds)
            )
            pooled_default_metrics = finalize_threshold_metrics(
                self._counts_to_dict(self.pooled_default_counts_by_size[int(pool_size)], [self.default_threshold])
            )[0]
            pooled_hourly_default_metrics = finalize_threshold_metrics(
                self._counts_to_dict(self.pooled_hourly_default_counts_by_size[int(pool_size)], [self.default_threshold])
            )[0]
            pooled_default_metrics_by_size[int(pool_size)] = pooled_default_metrics
            pooled_hourly_default_metrics_by_size[int(pool_size)] = pooled_hourly_default_metrics
            pooled_summaries_by_size[str(pool_size)] = {
                "best_threshold": float(pooled_overall["threshold"]),
                "best_csi": float(pooled_overall["csi"]),
                "best_pod": float(pooled_overall["pod"]),
                "best_far": float(pooled_overall["far"]),
                "threshold_sweep": pooled_threshold_sweep,
                "hourly_best_threshold": float(pooled_hourly_overall["threshold"]),
                "hourly_best_csi": float(pooled_hourly_overall["csi"]),
                "hourly_best_pod": float(pooled_hourly_overall["pod"]),
                "hourly_best_far": float(pooled_hourly_overall["far"]),
                "hourly_threshold_sweep": pooled_hourly_threshold_sweep,
                "default_csi": float(pooled_default_metrics["csi"]),
                "default_pod": float(pooled_default_metrics["pod"]),
                "default_far": float(pooled_default_metrics["far"]),
                "hourly_default_csi": float(pooled_hourly_default_metrics["csi"]),
                "hourly_default_pod": float(pooled_hourly_default_metrics["pod"]),
                "hourly_default_far": float(pooled_hourly_default_metrics["far"]),
            }
            per_lead_metrics_by_size[str(pool_size)] = build_per_lead_metrics_from_counts(
                [self._counts_to_dict(counts, self.thresholds) for counts in self.pooled_per_lead_counts_by_size[int(pool_size)]],
                float(pooled_overall["threshold"]),
            )

        primary_pooled_summary = pooled_summaries_by_size[str(self.primary_pool_size)]
        primary_pooled_default_metrics = pooled_default_metrics_by_size[int(self.primary_pool_size)]
        primary_pooled_hourly_default_metrics = pooled_hourly_default_metrics_by_size[int(self.primary_pool_size)]
        exact_per_lead = build_per_lead_metrics_from_counts(
            [self._counts_to_dict(counts, self.thresholds) for counts in self.per_lead_counts],
            float(overall["threshold"]),
        )
        # Full CSI matrix: rows = thresholds, columns = lead times.
        # Shape: [num_thresholds, num_lead_times]
        per_lead_csi_by_threshold = []

        for threshold_idx, threshold in enumerate(self.thresholds):
            row = []

            for lead_idx in range(len(self.per_lead_counts)):
                counts = self.per_lead_counts[lead_idx][threshold_idx]

                tp = float(counts[0].item())
                fp = float(counts[1].item())
                fn = float(counts[2].item())

                denominator = tp + fp + fn
                csi = tp / denominator if denominator > 0.0 else 0.0

                row.append(csi)

            per_lead_csi_by_threshold.append(row)

        eps = 1.0e-8
        pixel_count = max(float(self.pixel_count.item()), 1.0)
        brier = float(self.brier_sum.item() / pixel_count)
        clim = float(self.truth_sum.item() / pixel_count)
        bss = float(1.0 - brier / (clim * (1.0 - clim) + eps))
        return {
            "brier": brier,
            "bss": bss,
            "default_threshold": float(self.default_threshold),
            "pooled_pool_size": int(self.primary_pool_size),
            "pooled_pool_sizes": [int(value) for value in self.pooled_pool_sizes],
            "best_threshold": float(overall["threshold"]),
            "best_csi": float(overall["csi"]),
            "best_pod": float(overall["pod"]),
            "best_far": float(overall["far"]),
            "threshold_sweep": threshold_sweep,
            "pooled_best_threshold": float(primary_pooled_summary["best_threshold"]),
            "pooled_best_csi": float(primary_pooled_summary["best_csi"]),
            "pooled_best_pod": float(primary_pooled_summary["best_pod"]),
            "pooled_best_far": float(primary_pooled_summary["best_far"]),
            "pooled_threshold_sweep": primary_pooled_summary["threshold_sweep"],
            "hourly_best_threshold": float(hourly_overall["threshold"]),
            "hourly_best_csi": float(hourly_overall["csi"]),
            "hourly_best_pod": float(hourly_overall["pod"]),
            "hourly_best_far": float(hourly_overall["far"]),
            "hourly_threshold_sweep": hourly_threshold_sweep,
            "pooled_hourly_best_threshold": float(primary_pooled_summary["hourly_best_threshold"]),
            "pooled_hourly_best_csi": float(primary_pooled_summary["hourly_best_csi"]),
            "pooled_hourly_best_pod": float(primary_pooled_summary["hourly_best_pod"]),
            "pooled_hourly_best_far": float(primary_pooled_summary["hourly_best_far"]),
            "pooled_hourly_threshold_sweep": primary_pooled_summary["hourly_threshold_sweep"],
            "default_csi": float(default_metrics["csi"]),
            "default_pod": float(default_metrics["pod"]),
            "default_far": float(default_metrics["far"]),
            "pooled_default_csi": float(primary_pooled_default_metrics["csi"]),
            "pooled_default_pod": float(primary_pooled_default_metrics["pod"]),
            "pooled_default_far": float(primary_pooled_default_metrics["far"]),
            "hourly_default_csi": float(hourly_default_metrics["csi"]),
            "hourly_default_pod": float(hourly_default_metrics["pod"]),
            "hourly_default_far": float(hourly_default_metrics["far"]),
            "pooled_hourly_default_csi": float(primary_pooled_hourly_default_metrics["csi"]),
            "pooled_hourly_default_pod": float(primary_pooled_hourly_default_metrics["pod"]),
            "pooled_hourly_default_far": float(primary_pooled_hourly_default_metrics["far"]),
            "pooled_metrics_by_pool_size": pooled_summaries_by_size,
            "per_lead": exact_per_lead,
            "per_lead_csi_by_threshold": {
            "thresholds": [float(value) for value in self.thresholds],
            "lead_times": [int((lead_idx + 1) * 5)for lead_idx in range(len(self.per_lead_counts))],
            "csi": per_lead_csi_by_threshold,
},
            "pooled_per_lead": per_lead_metrics_by_size[str(self.primary_pool_size)],
            "pooled_per_lead_by_pool_size": per_lead_metrics_by_size,
        }


def rank_sharded_export_path(path: Path, rank: int, world_size: int) -> Path:
    if world_size <= 1:
        return path
    return path.with_name(f"{path.stem}.rank{rank:02d}-of-{world_size:02d}{path.suffix}")


def create_branch_attention_export(
    export_path: Path,
    sample_count: int,
    lead_time: int,
    height: int,
    width: int,
    future_condition_mode: str,
) -> tuple[h5py.File, dict[str, h5py.Dataset]]:
    export_path.parent.mkdir(parents=True, exist_ok=True)
    h5_file = h5py.File(export_path, "w")
    h5_file.attrs["future_condition_mode"] = str(future_condition_mode)
    h5_file.attrs["branch_order"] = json.dumps(["low", "mid", "high"])

    datasets = {
        "branch_attention": h5_file.create_dataset(
            "branch_attention",
            shape=(sample_count, lead_time, 3, height, width),
            dtype=np.float16,
            compression="gzip",
            chunks=(1, lead_time, 3, height, width),
        ),
        "branch_attention_logits": h5_file.create_dataset(
            "branch_attention_logits",
            shape=(sample_count, lead_time, 3, height, width),
            dtype=np.float16,
            compression="gzip",
            chunks=(1, lead_time, 3, height, width),
        ),
        "branch_inputs": h5_file.create_dataset(
            "branch_inputs",
            shape=(sample_count, lead_time, 3, height, width),
            dtype=np.float16,
            compression="gzip",
            chunks=(1, lead_time, 3, height, width),
        ),
        "branch_spread": h5_file.create_dataset(
            "branch_spread",
            shape=(sample_count, lead_time, height, width),
            dtype=np.float16,
            compression="gzip",
            chunks=(1, lead_time, height, width),
        ),
        "predicted_probability": h5_file.create_dataset(
            "predicted_probability",
            shape=(sample_count, lead_time, height, width),
            dtype=np.float16,
            compression="gzip",
            chunks=(1, lead_time, height, width),
        ),
        "target_lightning": h5_file.create_dataset(
            "target_lightning",
            shape=(sample_count, lead_time, height, width),
            dtype=np.float16,
            compression="gzip",
            chunks=(1, lead_time, height, width),
        ),
        "sample_id": h5_file.create_dataset(
            "sample_id",
            shape=(sample_count,),
            dtype=h5py.string_dtype(encoding="utf-8"),
        ),
    }
    return h5_file, datasets


def create_prediction_export(
    export_path: Path,
    sample_count: int,
    lead_time: int,
    height: int,
    width: int,
    include_uncertainty: bool,
) -> tuple[h5py.File, dict[str, h5py.Dataset]]:
    export_path.parent.mkdir(parents=True, exist_ok=True)
    h5_file = h5py.File(export_path, "w")
    h5_file.attrs["include_uncertainty"] = bool(include_uncertainty)

    datasets = {
        "predicted_probability": h5_file.create_dataset(
            "predicted_probability",
            shape=(sample_count, lead_time, height, width),
            dtype=np.float16,
            compression="gzip",
            chunks=(1, lead_time, height, width),
        ),
        "target_lightning": h5_file.create_dataset(
            "target_lightning",
            shape=(sample_count, lead_time, height, width),
            dtype=np.float16,
            compression="gzip",
            chunks=(1, lead_time, height, width),
        ),
        "sample_id": h5_file.create_dataset(
            "sample_id",
            shape=(sample_count,),
            dtype=h5py.string_dtype(encoding="utf-8"),
        ),
    }
    if include_uncertainty:
        datasets["predicted_uncertainty"] = h5_file.create_dataset(
            "predicted_uncertainty",
            shape=(sample_count, lead_time, height, width),
            dtype=np.float16,
            compression="gzip",
            chunks=(1, lead_time, height, width),
        )
        datasets["predicted_variance"] = h5_file.create_dataset(
            "predicted_variance",
            shape=(sample_count, lead_time, height, width),
            dtype=np.float16,
            compression="gzip",
            chunks=(1, lead_time, height, width),
        )
    return h5_file, datasets


def write_prediction_batch(
    datasets: dict[str, h5py.Dataset],
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    sample_ids: list[str],
    write_index: int,
) -> int:
    batch_len = int(target.shape[0])
    next_index = write_index + batch_len
    datasets["predicted_probability"][write_index:next_index] = (
        outputs["probs"].squeeze(2).detach().cpu().numpy().astype(np.float16, copy=False)
    )
    if "predicted_uncertainty" in datasets:
        if "uncertainty" not in outputs or "variance" not in outputs:
            raise ValueError(
                "Requested prediction export with uncertainty datasets, but model outputs do not include uncertainty and variance. "
                "Use a checkpoint trained with model.predict_uncertainty=true."
            )
        datasets["predicted_uncertainty"][write_index:next_index] = (
            outputs["uncertainty"].squeeze(2).detach().cpu().numpy().astype(np.float16, copy=False)
        )
        datasets["predicted_variance"][write_index:next_index] = (
            outputs["variance"].squeeze(2).detach().cpu().numpy().astype(np.float16, copy=False)
        )
    datasets["target_lightning"][write_index:next_index] = (
        target.squeeze(2).detach().cpu().numpy().astype(np.float16, copy=False)
    )
    datasets["sample_id"][write_index:next_index] = list(sample_ids)
    return next_index


def write_branch_attention_batch(
    datasets: dict[str, h5py.Dataset],
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    sample_ids: list[str],
    write_index: int,
) -> int:
    required_keys = {
        "branch_attention",
        "branch_attention_logits",
        "branch_inputs",
        "branch_spread",
    }
    missing = sorted(required_keys.difference(outputs))
    if missing:
        raise ValueError(
            "Requested branch-attention export, but model outputs are missing required tensors: "
            f"{', '.join(missing)}. Use future_condition_mode='interval_attention'."
        )

    batch_len = int(target.shape[0])
    next_index = write_index + batch_len
    datasets["branch_attention"][write_index:next_index] = (
        outputs["branch_attention"].detach().cpu().numpy().astype(np.float16, copy=False)
    )
    datasets["branch_attention_logits"][write_index:next_index] = (
        outputs["branch_attention_logits"].detach().cpu().numpy().astype(np.float16, copy=False)
    )
    datasets["branch_inputs"][write_index:next_index] = (
        outputs["branch_inputs"].squeeze(3).detach().cpu().numpy().astype(np.float16, copy=False)
    )
    datasets["branch_spread"][write_index:next_index] = (
        outputs["branch_spread"].squeeze(2).detach().cpu().numpy().astype(np.float16, copy=False)
    )
    datasets["predicted_probability"][write_index:next_index] = (
        outputs["probs"].squeeze(2).detach().cpu().numpy().astype(np.float16, copy=False)
    )
    datasets["target_lightning"][write_index:next_index] = (
        target.squeeze(2).detach().cpu().numpy().astype(np.float16, copy=False)
    )
    datasets["sample_id"][write_index:next_index] = list(sample_ids)
    return next_index


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank, device = setup_ddp(args)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    target_hw = tuple(config["model"]["target_hw"])

    if rank == 0:
        print(
            json.dumps(
                {
                    "event": "eval_init",
                    "split": args.split,
                    "world_size": world_size,
                    "device": str(device),
                    "checkpoint": str(args.checkpoint),
                    "prepare_spatial_mask_only": bool(args.prepare_spatial_mask_only),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    model = None
    if not args.prepare_spatial_mask_only:
        model = LightningSimVPSystem(**config["model"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()

    dataset = build_dataset(args.split, config["data"])
    dataset = maybe_wrap_dataset_with_future_condition(dataset, config, args.split)
    precomputed_mask, precomputed_mask_source = resolve_precomputed_spatial_mask(
        config,
        args.checkpoint,
        args.split,
        target_hw,
    )
    mask_cache_path = build_spatial_mask_cache_path(config["data"], target_hw)
    spatial_mask, mask_bbox, mask_source = infer_and_sync_spatial_mask(
        dataset,
        target_hw,
        rank,
        device,
        precomputed_mask=precomputed_mask,
        precomputed_mask_source=precomputed_mask_source,
        cache_path=mask_cache_path,
    )
    if rank == 0:
        print(
            json.dumps(
                {
                    "event": "spatial_mask_ready",
                    "split": args.split,
                    "source": mask_source,
                    "cache_path": str(mask_cache_path),
                    "pixel_count": int(spatial_mask.sum()),
                    "bbox": mask_bbox,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if args.prepare_spatial_mask_only:
        if hasattr(dataset, "close"):
            dataset.close()
        cleanup_ddp()
        return

    local_dataset = shard_dataset_across_ranks(dataset, rank, world_size)
    dataloader = build_dataloader(
        local_dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        num_workers=int(config["evaluation"]["num_workers"]),
    )
    total_local_batches = len(dataloader)
    total_local_samples = len(local_dataset)

    future_condition_source = None
    radar_condition_config = config.get("radar_condition", {})
    if radar_condition_config.get("enabled", False) and str(radar_condition_config.get("source", "online")).lower() == "online":
        future_condition_source = build_future_condition_source(config, args.split, device)
        load_future_condition_source_state(future_condition_source, checkpoint)
        if future_condition_source is not None:
            future_condition_source.eval()
    elif checkpoint.get("future_condition_source_state_dict") is not None:
        raise ValueError(
            "Checkpoint includes future_condition_source_state_dict, but radar_condition.source is not 'online' for evaluation."
        )

    loss_fn = build_loss(
        config["training"]["loss_name"],
        float(config["training"]["focal_alpha"]),
        float(config["training"]["focal_gamma"]),
        config["training"].get("pos_weight"),
    )
    thresholds = [float(value) for value in config["evaluation"]["thresholds"]]
    default_threshold = float(config["evaluation"].get("default_threshold", 0.5))
    pooled_pool_sizes = [int(value) for value in config["evaluation"].get("pooled_pool_sizes", [2, 6, 12])]
    pooled_pool_sizes = list(dict.fromkeys(pooled_pool_sizes))
    if 2 not in pooled_pool_sizes:
        pooled_pool_sizes.insert(0, 2)

    total_loss_sum = 0.0
    total_examples = 0
    metric_accumulator = DeviceEvaluationMetricAccumulator(
        thresholds=thresholds,
        default_threshold=default_threshold,
        pooled_pool_sizes=pooled_pool_sizes,
        spatial_mask=spatial_mask,
        device=device,
    )

    branch_export_file = None
    branch_export_datasets = None
    branch_export_path = None
    branch_export_index = 0
    prediction_export_file = None
    prediction_export_datasets = None
    prediction_export_path = None
    prediction_export_index = 0
    eval_start_time = time.time()
    local_samples_seen = 0
    local_running_loss = 0.0
    if args.export_branch_attention:
        branch_export_path = args.branch_attention_output
        if branch_export_path is None:
            branch_export_path = args.checkpoint.parent / f"branch_attention_{args.split}.h5"
        branch_export_path = rank_sharded_export_path(branch_export_path, rank, world_size)
        branch_export_file, branch_export_datasets = create_branch_attention_export(
            export_path=branch_export_path,
            sample_count=len(local_dataset),
            lead_time=int(config["model"]["lead_time"]),
            height=int(config["model"]["target_hw"][0]),
            width=int(config["model"]["target_hw"][1]),
            future_condition_mode=str(config["model"].get("future_condition_mode", "simple")),
        )
        branch_export_file.attrs["split"] = args.split
        branch_export_file.attrs["checkpoint"] = str(args.checkpoint)
        branch_export_file.attrs["rank"] = int(rank)
        branch_export_file.attrs["world_size"] = int(world_size)
    if args.export_predictions:
        prediction_export_path = args.prediction_output
        if prediction_export_path is None:
            prediction_export_path = args.checkpoint.parent / f"predictions_{args.split}.h5"
        prediction_export_path = rank_sharded_export_path(prediction_export_path, rank, world_size)
        prediction_export_file, prediction_export_datasets = create_prediction_export(
            export_path=prediction_export_path,
            sample_count=len(local_dataset),
            lead_time=int(config["model"]["lead_time"]),
            height=int(config["model"]["target_hw"][0]),
            width=int(config["model"]["target_hw"][1]),
            include_uncertainty=bool(config["model"].get("predict_uncertainty", False)),
        )
        prediction_export_file.attrs["split"] = args.split
        prediction_export_file.attrs["checkpoint"] = str(args.checkpoint)
        prediction_export_file.attrs["rank"] = int(rank)
        prediction_export_file.attrs["world_size"] = int(world_size)

    try:
        if rank == 0:
            print(
                json.dumps(
                    {
                        "event": "eval_start",
                        "split": args.split,
                        "world_size": world_size,
                        "device": str(device),
                        "global_samples": len(dataset),
                        "local_samples": total_local_samples,
                        "local_batches": total_local_batches,
                        "progress_every": int(args.progress_every),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader, start=1):
                batch = move_batch(batch, device)
                future_condition = prepare_future_condition(batch, future_condition_source)
                outputs = model.predict_distribution(
                    batch["radar_past"],
                    batch["lightning_past"],
                    future_condition=future_condition,
                    time_emb=batch.get("time_emb"),
                )
                target = batch["lightning_future"].permute(0, 1, 4, 2, 3).contiguous()
                loss = loss_fn(outputs["logits"], target)
                batch_size = int(target.shape[0])
                total_loss_sum += float(loss.item()) * batch_size
                total_examples += batch_size
                local_samples_seen += batch_size
                local_running_loss += float(loss.item()) * batch_size
                metric_accumulator.update(outputs["probs"], target)

                if rank == 0 and int(args.progress_every) > 0:
                    if batch_idx % int(args.progress_every) == 0 or batch_idx == total_local_batches:
                        elapsed = time.time() - eval_start_time
                        mean_loss = local_running_loss / max(local_samples_seen, 1)
                        print(
                            json.dumps(
                                {
                                    "event": "eval_progress",
                                    "split": args.split,
                                    "batch_idx": batch_idx,
                                    "local_batches": total_local_batches,
                                    "local_samples_seen": local_samples_seen,
                                    "local_samples": total_local_samples,
                                    "elapsed_seconds": elapsed,
                                    "mean_loss_local": mean_loss,
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )

                if branch_export_datasets is not None:
                    branch_export_index = write_branch_attention_batch(
                        branch_export_datasets,
                        outputs,
                        target,
                        sample_ids=list(batch["sample_id"]),
                        write_index=branch_export_index,
                    )
                if prediction_export_datasets is not None:
                    prediction_export_index = write_prediction_batch(
                        prediction_export_datasets,
                        outputs,
                        target,
                        sample_ids=list(batch["sample_id"]),
                        write_index=prediction_export_index,
                    )
    finally:
        if branch_export_file is not None:
            branch_export_file.close()
        if prediction_export_file is not None:
            prediction_export_file.close()

    sync_device(device)
    summary = metric_accumulator.finalize()
    reduced_loss = reduce_loss_sum_and_count(total_loss_sum, total_examples, device)
    metrics = {
        "split": args.split,
        "loss": reduced_loss,
        "spatial_mask_bbox": mask_bbox,
        "spatial_mask_pixel_count": int(spatial_mask.sum()),
        **summary,
    }
    if args.export_branch_attention:
        if world_size > 1:
            base_export_path = args.branch_attention_output or (args.checkpoint.parent / f"branch_attention_{args.split}.h5")
            metrics["branch_attention_export_paths"] = [
                str(rank_sharded_export_path(base_export_path, export_rank, world_size)) for export_rank in range(world_size)
            ]
        else:
            metrics["branch_attention_export_path"] = str(branch_export_path)
    if args.export_predictions:
        if world_size > 1:
            base_export_path = args.prediction_output or (args.checkpoint.parent / f"predictions_{args.split}.h5")
            metrics["prediction_export_paths"] = [
                str(rank_sharded_export_path(base_export_path, export_rank, world_size)) for export_rank in range(world_size)
            ]
        else:
            metrics["prediction_export_path"] = str(prediction_export_path)

    if dist.is_initialized():
        dist.barrier(device_ids=[local_rank])

    if rank == 0:
        output_path = args.checkpoint.parent / f"metrics_{args.split}.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        print(json.dumps(metrics, sort_keys=True))

    if dist.is_initialized():
        dist.barrier(device_ids=[local_rank])

    if hasattr(dataset, "close"):
        dataset.close()
    if future_condition_source is not None:
        future_condition_source.close()
    cleanup_ddp()


if __name__ == "__main__":
    main()
