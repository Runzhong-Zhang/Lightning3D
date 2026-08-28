from __future__ import annotations

from typing import Any

import numpy as np
import torch


def binary_brier_score(probs: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean((probs.float() - target.float()) ** 2).item())


def csi_counts(probs: torch.Tensor, target: torch.Tensor, threshold: float) -> tuple[float, float, float]:
    pred = probs >= float(threshold)
    truth = target >= 0.5
    tp = float((pred & truth).sum().item())
    fp = float((pred & (~truth)).sum().item())
    fn = float(((~pred) & truth).sum().item())
    return tp, fp, fn


def csi_from_counts(tp: float, fp: float, fn: float) -> float:
    denom = tp + fp + fn
    if denom <= 0.0:
        return 0.0
    return tp / denom


def best_csi_threshold(
    probs: torch.Tensor,
    target: torch.Tensor,
    thresholds: list[float],
) -> tuple[float, float]:
    best_threshold = float(thresholds[0])
    best_csi = -1.0
    for threshold in thresholds:
        tp, fp, fn = csi_counts(probs, target, threshold)
        csi = csi_from_counts(tp, fp, fn)
        if csi > best_csi:
            best_threshold = float(threshold)
            best_csi = float(csi)
    return best_threshold, best_csi


def _to_numpy_spatiotemporal(array: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().float().numpy()
    else:
        array = np.asarray(array, dtype=np.float32)

    if array.ndim == 5:
        if array.shape[2] == 1:
            array = array[:, :, 0, :, :]
        elif array.shape[-1] == 1:
            array = array[..., 0]
        else:
            raise ValueError(f"Expected single-channel array, got shape {array.shape}")
    if array.ndim != 4:
        raise ValueError(f"Expected shape (B, T, H, W) after squeeze, got {array.shape}")
    return array.astype(np.float32, copy=False)


def _normalize_spatial_mask(spatial_mask: np.ndarray | None, spatial_shape: tuple[int, int]) -> np.ndarray | None:
    if spatial_mask is None:
        return None
    mask = np.asarray(spatial_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"Expected spatial_mask with shape (H, W), got {mask.shape}")
    if mask.shape != spatial_shape:
        raise ValueError(f"Expected spatial_mask shape {spatial_shape}, got {mask.shape}")
    return mask


def _broadcast_spatial_mask(spatial_mask: np.ndarray | None, array: np.ndarray) -> np.ndarray | None:
    mask = _normalize_spatial_mask(spatial_mask, array.shape[-2:])
    if mask is None:
        return None
    return np.broadcast_to(mask, array.shape)


def max_pool_spatial(arr: np.ndarray, pool_size: int) -> np.ndarray:
    if pool_size <= 1:
        return arr
    if arr.ndim < 2:
        raise ValueError("Expected at least 2 spatial dimensions for max pooling.")

    h, w = arr.shape[-2], arr.shape[-1]
    pad_h = (-h) % pool_size
    pad_w = (-w) % pool_size
    if pad_h or pad_w:
        pad_width = [(0, 0)] * arr.ndim
        pad_width[-2] = (0, pad_h)
        pad_width[-1] = (0, pad_w)
        arr = np.pad(arr, pad_width, mode="constant", constant_values=0)
        h, w = arr.shape[-2], arr.shape[-1]

    new_shape = arr.shape[:-2] + (h // pool_size, pool_size, w // pool_size, pool_size)
    return arr.reshape(new_shape).max(axis=(-3, -1))


def compute_binary_metrics(
    scores: np.ndarray,
    truth: np.ndarray,
    threshold: float,
    pool_size: int = 1,
    spatial_mask: np.ndarray | None = None,
) -> dict[str, float]:
    pred = scores >= float(threshold)
    true = truth >= 0.5
    valid = _broadcast_spatial_mask(spatial_mask, pred)
    if valid is not None:
        pred = np.logical_and(pred, valid)
        true = np.logical_and(true, valid)
    pred = max_pool_spatial(pred, pool_size)
    true = max_pool_spatial(true, pool_size)
    pooled_valid = None if valid is None else max_pool_spatial(valid, pool_size).astype(bool, copy=False)

    tp_mask = np.logical_and(pred, true)
    fp_mask = np.logical_and(pred, np.logical_not(true))
    fn_mask = np.logical_and(np.logical_not(pred), true)
    tn_mask = np.logical_and(np.logical_not(pred), np.logical_not(true))
    if pooled_valid is not None:
        tp_mask = np.logical_and(tp_mask, pooled_valid)
        fp_mask = np.logical_and(fp_mask, pooled_valid)
        fn_mask = np.logical_and(fn_mask, pooled_valid)
        tn_mask = np.logical_and(tn_mask, pooled_valid)

    tp = tp_mask.sum()
    fp = fp_mask.sum()
    fn = fn_mask.sum()
    tn = tn_mask.sum()

    eps = 1.0e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    far = fp / (tp + fp + eps)
    csi = tp / (tp + fp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision": float(precision),
        "recall": float(recall),
        "pod": float(recall),
        "far": float(far),
        "csi": float(csi),
        "f1": float(f1),
    }


def compute_probabilistic_metrics(probs: np.ndarray, truth: np.ndarray, spatial_mask: np.ndarray | None = None) -> dict[str, float]:
    true = truth >= 0.5
    valid = _broadcast_spatial_mask(spatial_mask, probs)
    if valid is not None:
        probs = probs[valid]
        true = true[valid]
    if probs.size == 0:
        raise ValueError("Spatial mask removed all pixels from probabilistic metric computation.")
    eps = 1.0e-8
    brier = np.mean((probs.astype(np.float64) - true.astype(np.float64)) ** 2)
    clim = true.mean()
    brier_ref = float(clim * (1.0 - clim))
    bss = 1.0 - brier / (brier_ref + eps)
    return {
        "brier": float(brier),
        "bss": float(bss),
    }


def init_threshold_counts(thresholds: list[float]) -> dict[float, dict[str, float]]:
    return {float(threshold): {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0} for threshold in thresholds}


def update_threshold_counts(
    metric_counts: dict[float, dict[str, float]],
    scores: np.ndarray,
    truth: np.ndarray,
    pool_size: int = 1,
    spatial_mask: np.ndarray | None = None,
) -> None:
    truth_mask = truth >= 0.5
    valid = _broadcast_spatial_mask(spatial_mask, truth_mask)
    for threshold, counts in metric_counts.items():
        pred_mask = scores >= float(threshold)
        if valid is not None:
            pred_mask = np.logical_and(pred_mask, valid)
            truth_mask_local = np.logical_and(truth_mask, valid)
        else:
            truth_mask_local = truth_mask
        pooled_pred = max_pool_spatial(pred_mask, pool_size)
        pooled_truth = max_pool_spatial(truth_mask_local, pool_size)
        pooled_valid = None if valid is None else max_pool_spatial(valid, pool_size).astype(bool, copy=False)
        tp_mask = np.logical_and(pooled_pred, pooled_truth)
        fp_mask = np.logical_and(pooled_pred, np.logical_not(pooled_truth))
        fn_mask = np.logical_and(np.logical_not(pooled_pred), pooled_truth)
        tn_mask = np.logical_and(np.logical_not(pooled_pred), np.logical_not(pooled_truth))
        if pooled_valid is not None:
            tp_mask = np.logical_and(tp_mask, pooled_valid)
            fp_mask = np.logical_and(fp_mask, pooled_valid)
            fn_mask = np.logical_and(fn_mask, pooled_valid)
            tn_mask = np.logical_and(tn_mask, pooled_valid)
        counts["tp"] += float(tp_mask.sum())
        counts["fp"] += float(fp_mask.sum())
        counts["fn"] += float(fn_mask.sum())
        counts["tn"] += float(tn_mask.sum())


def finalize_threshold_metrics(metric_counts: dict[float, dict[str, float]]) -> tuple[dict[str, float], list[dict[str, float]]]:
    best_metrics: dict[str, float] | None = None
    threshold_metrics = []
    eps = 1.0e-8
    for threshold, counts in metric_counts.items():
        tp = float(counts["tp"])
        fp = float(counts["fp"])
        fn = float(counts["fn"])
        tn = float(counts["tn"])
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        far = fp / (tp + fp + eps)
        csi = tp / (tp + fp + fn + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        metrics = {
            "threshold": float(threshold),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": float(precision),
            "pod": float(recall),
            "far": float(far),
            "csi": float(csi),
            "f1": float(f1),
        }
        threshold_metrics.append(metrics)
        if best_metrics is None or metrics["csi"] > best_metrics["csi"]:
            best_metrics = metrics
    if best_metrics is None:
        raise ValueError("No thresholds available to finalize metrics.")
    return best_metrics, threshold_metrics


def build_per_lead_metrics_from_counts(
    per_lead_counts: list[dict[float, dict[str, float]]],
    threshold: float,
) -> list[dict[str, float]]:
    lead_metrics = []
    eps = 1.0e-8
    for lead_idx, counts_for_threshold in enumerate(per_lead_counts):
        counts = counts_for_threshold[float(threshold)]
        tp = float(counts["tp"])
        fp = float(counts["fp"])
        fn = float(counts["fn"])
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        lead_metrics.append(
            {
                "lead_idx": int(lead_idx),
                "csi": float(tp / (tp + fp + fn + eps)),
                "pod": float(recall),
                "far": float(fp / (tp + fp + eps)),
                "precision": float(precision),
                "f1": float(2.0 * precision * recall / (precision + recall + eps)),
            }
        )
    return lead_metrics


class StreamingForecastMetricAccumulator:
    def __init__(
        self,
        thresholds: list[float],
        default_threshold: float = 0.5,
        pooled_pool_size: int = 2,
        include_details: bool = False,
        spatial_mask: np.ndarray | None = None,
    ):
        self.thresholds = [float(value) for value in thresholds]
        self.default_threshold = float(default_threshold)
        self.pooled_pool_size = int(pooled_pool_size)
        self.include_details = bool(include_details)
        self.spatial_mask = None if spatial_mask is None else np.asarray(spatial_mask, dtype=bool)

        self.frame_counts = init_threshold_counts(self.thresholds)
        self.pooled_frame_counts = init_threshold_counts(self.thresholds)
        self.hourly_counts = init_threshold_counts(self.thresholds)
        self.pooled_hourly_counts = init_threshold_counts(self.thresholds)

        self.default_counts = init_threshold_counts([self.default_threshold])
        self.pooled_default_counts = init_threshold_counts([self.default_threshold])
        self.hourly_default_counts = init_threshold_counts([self.default_threshold])
        self.pooled_hourly_default_counts = init_threshold_counts([self.default_threshold])

        self.per_lead_counts: list[dict[float, dict[str, float]]] | None = None
        self.pooled_per_lead_counts: list[dict[float, dict[str, float]]] | None = None

        self.brier_sum = 0.0
        self.truth_sum = 0.0
        self.pixel_count = 0

    def update(self, probs: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray) -> None:
        mean_raw_forecast = _to_numpy_spatiotemporal(probs)
        truth_array = _to_numpy_spatiotemporal(target)
        mean_probs = np.clip(mean_raw_forecast, 0.0, 1.0)
        spatial_mask = _normalize_spatial_mask(self.spatial_mask, mean_raw_forecast.shape[-2:])
        truth_binary = truth_array >= 0.5
        valid = _broadcast_spatial_mask(spatial_mask, mean_probs)
        brier_error = mean_probs.astype(np.float64) - truth_binary.astype(np.float64)

        if valid is not None:
            self.brier_sum += float(np.square(brier_error)[valid].sum())
            self.truth_sum += float(truth_binary[valid].sum())
            self.pixel_count += int(valid.sum())
        else:
            self.brier_sum += float(np.square(brier_error).sum())
            self.truth_sum += float(truth_binary.sum())
            self.pixel_count += int(truth_binary.size)

        update_threshold_counts(self.frame_counts, mean_raw_forecast, truth_array, pool_size=1, spatial_mask=spatial_mask)
        update_threshold_counts(
            self.pooled_frame_counts,
            mean_raw_forecast,
            truth_array,
            pool_size=self.pooled_pool_size,
            spatial_mask=spatial_mask,
        )
        update_threshold_counts(self.default_counts, mean_raw_forecast, truth_array, pool_size=1, spatial_mask=spatial_mask)
        update_threshold_counts(
            self.pooled_default_counts,
            mean_raw_forecast,
            truth_array,
            pool_size=self.pooled_pool_size,
            spatial_mask=spatial_mask,
        )

        hourly_truth = aggregate_hourly_binary_target(truth_array)
        hourly_score = aggregate_hourly_forecast_score(mean_raw_forecast)
        update_threshold_counts(self.hourly_counts, hourly_score, hourly_truth, pool_size=1, spatial_mask=spatial_mask)
        update_threshold_counts(
            self.pooled_hourly_counts,
            hourly_score,
            hourly_truth,
            pool_size=self.pooled_pool_size,
            spatial_mask=spatial_mask,
        )
        update_threshold_counts(self.hourly_default_counts, hourly_score, hourly_truth, pool_size=1, spatial_mask=spatial_mask)
        update_threshold_counts(
            self.pooled_hourly_default_counts,
            hourly_score,
            hourly_truth,
            pool_size=self.pooled_pool_size,
            spatial_mask=spatial_mask,
        )

        if self.include_details:
            if self.per_lead_counts is None:
                self.per_lead_counts = [init_threshold_counts(self.thresholds) for _ in range(mean_raw_forecast.shape[1])]
                self.pooled_per_lead_counts = [
                    init_threshold_counts(self.thresholds) for _ in range(mean_raw_forecast.shape[1])
                ]
            assert self.pooled_per_lead_counts is not None
            for lead_idx in range(mean_raw_forecast.shape[1]):
                update_threshold_counts(
                    self.per_lead_counts[lead_idx],
                    mean_raw_forecast[:, lead_idx],
                    truth_array[:, lead_idx],
                    pool_size=1,
                    spatial_mask=spatial_mask,
                )
                update_threshold_counts(
                    self.pooled_per_lead_counts[lead_idx],
                    mean_raw_forecast[:, lead_idx],
                    truth_array[:, lead_idx],
                    pool_size=self.pooled_pool_size,
                    spatial_mask=spatial_mask,
                )

    def finalize(self) -> dict[str, Any]:
        overall, threshold_sweep = finalize_threshold_metrics(self.frame_counts)
        pooled_overall, pooled_threshold_sweep = finalize_threshold_metrics(self.pooled_frame_counts)
        hourly_overall, hourly_threshold_sweep = finalize_threshold_metrics(self.hourly_counts)
        pooled_hourly_overall, pooled_hourly_threshold_sweep = finalize_threshold_metrics(self.pooled_hourly_counts)

        default_metrics, _ = finalize_threshold_metrics(self.default_counts)
        pooled_default_metrics, _ = finalize_threshold_metrics(self.pooled_default_counts)
        hourly_default_metrics, _ = finalize_threshold_metrics(self.hourly_default_counts)
        pooled_hourly_default_metrics, _ = finalize_threshold_metrics(self.pooled_hourly_default_counts)

        eps = 1.0e-8
        brier = float(self.brier_sum / max(self.pixel_count, 1))
        clim = float(self.truth_sum / max(self.pixel_count, 1))
        bss = float(1.0 - brier / (clim * (1.0 - clim) + eps))

        summary: dict[str, Any] = {
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

        if self.include_details:
            summary["threshold_sweep"] = threshold_sweep
            summary["pooled_threshold_sweep"] = pooled_threshold_sweep
            summary["hourly_threshold_sweep"] = hourly_threshold_sweep
            summary["pooled_hourly_threshold_sweep"] = pooled_hourly_threshold_sweep
            if self.per_lead_counts is not None and self.pooled_per_lead_counts is not None:
                summary["per_lead"] = build_per_lead_metrics_from_counts(
                    self.per_lead_counts,
                    float(overall["threshold"]),
                )
                summary["pooled_per_lead"] = build_per_lead_metrics_from_counts(
                    self.pooled_per_lead_counts,
                    float(pooled_overall["threshold"]),
                )
        return summary


def aggregate_hourly_binary_target(truth: np.ndarray) -> np.ndarray:
    return (truth >= 0.5).max(axis=1).astype(np.float32)


def aggregate_hourly_forecast_score(mean_raw_forecast: np.ndarray) -> np.ndarray:
    return mean_raw_forecast.max(axis=1)


def find_best_csi_threshold(
    probs: np.ndarray,
    truth: np.ndarray,
    thresholds: list[float],
    pool_size: int = 1,
    spatial_mask: np.ndarray | None = None,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    sweep_results = []
    best_metrics = None
    for threshold in thresholds:
        metrics = compute_binary_metrics(probs, truth, float(threshold), pool_size=pool_size, spatial_mask=spatial_mask)
        metrics["threshold"] = float(threshold)
        sweep_results.append(metrics)
        if best_metrics is None or metrics["csi"] > best_metrics["csi"]:
            best_metrics = metrics
    if best_metrics is None:
        raise ValueError("No thresholds were provided.")
    return best_metrics, sweep_results


def compute_per_lead_metrics(
    probs: np.ndarray,
    truth: np.ndarray,
    threshold: float,
    pool_size: int = 1,
    spatial_mask: np.ndarray | None = None,
) -> list[dict[str, float]]:
    lead_metrics = []
    for lead_idx in range(probs.shape[1]):
        metrics = compute_binary_metrics(
            probs[:, lead_idx],
            truth[:, lead_idx],
            threshold,
            pool_size=pool_size,
            spatial_mask=spatial_mask,
        )
        metrics["lead_idx"] = int(lead_idx)
        lead_metrics.append(metrics)
    return lead_metrics


def summarize_forecast_metrics(
    probs: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    thresholds: list[float],
    default_threshold: float = 0.5,
    pooled_pool_size: int = 2,
    include_details: bool = False,
    spatial_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    accumulator = StreamingForecastMetricAccumulator(
        thresholds=thresholds,
        default_threshold=default_threshold,
        pooled_pool_size=pooled_pool_size,
        include_details=include_details,
        spatial_mask=spatial_mask,
    )
    accumulator.update(probs, target)
    return accumulator.finalize()