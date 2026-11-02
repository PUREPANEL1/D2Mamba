"""Lesion metrics for DDR evaluation."""

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.evaluator import BaseMetric
from mmengine.logging import print_log
from mmseg.registry import METRICS


LESION_NAMES = (
    "EX",
    "MA",
    "SE",
    "HE",
)


@METRICS.register_module()
class DDRLesionMetric(BaseMetric):
    """DDR lesion IoU, Dice, and binned AUPR metric."""

    def __init__(
        self,
        num_classes=5,
        ignore_index=255,
        aupr_bins=1000,
        collect_device="cpu",
        prefix=None,
    ):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.aupr_bins = aupr_bins

    def process(self, data_batch, data_samples):
        for data_sample in data_samples:
            meta = (data_sample.metainfo if hasattr(data_sample, "metainfo")
                    else data_sample)
            if meta.get("flip", False):
                raise ValueError(
                    "DDR evaluation received flip=True. MMSeg unflips predictions "
                    "but training RandomFlip also flips GT. Use the deterministic "
                    "val/test pipeline, not the training augmentation pipeline.")
            pred = self._get_label_map(data_sample, "pred_sem_seg")
            target = self._get_label_map(data_sample, "gt_sem_seg")

            if pred.shape != target.shape:
                raise ValueError(
                    f"Prediction/GT shape mismatch: {tuple(pred.shape)} vs "
                    f"{tuple(target.shape)} for {meta.get('img_path', '<unknown>')}. "
                    "Check preprocessing ori_shape and annotation coordinates; "
                    "the metric must not resize class predictions.")

            score = self._extract_score(data_sample, target.shape)
            valid = target != self.ignore_index
            pred = pred[valid]
            target = target[valid]
            score = score[:, valid]

            intersect = torch.zeros(self.num_classes, dtype=torch.float64)
            pred_area = torch.zeros(self.num_classes, dtype=torch.float64)
            target_area = torch.zeros(self.num_classes, dtype=torch.float64)
            pos_hist = torch.zeros((self.num_classes, self.aupr_bins), dtype=torch.float64)
            total_hist = torch.zeros((self.num_classes, self.aupr_bins), dtype=torch.float64)

            for class_idx in range(self.num_classes):
                pred_i = pred == class_idx
                target_i = target == class_idx
                intersect[class_idx] = (pred_i & target_i).sum()
                pred_area[class_idx] = pred_i.sum()
                target_area[class_idx] = target_i.sum()

                bin_idx = torch.clamp(
                    (score[class_idx] * (self.aupr_bins - 1)).to(torch.long),
                    min=0,
                    max=self.aupr_bins - 1,
                )
                ones = torch.ones_like(bin_idx, dtype=torch.float64)
                total_hist[class_idx].scatter_add_(0, bin_idx, ones)
                pos_hist[class_idx].scatter_add_(0, bin_idx, target_i.to(torch.float64))

            self.results.append(
                dict(
                    intersect=intersect.numpy(),
                    pred_area=pred_area.numpy(),
                    target_area=target_area.numpy(),
                    pos_hist=pos_hist.numpy(),
                    total_hist=total_hist.numpy(),
                )
            )

    def compute_metrics(self, results):
        intersect = sum((item["intersect"] for item in results), np.zeros(self.num_classes))
        pred_area = sum((item["pred_area"] for item in results), np.zeros(self.num_classes))
        target_area = sum((item["target_area"] for item in results), np.zeros(self.num_classes))
        pos_hist = sum(
            (item["pos_hist"] for item in results),
            np.zeros((self.num_classes, self.aupr_bins)),
        )
        total_hist = sum(
            (item["total_hist"] for item in results),
            np.zeros((self.num_classes, self.aupr_bins)),
        )

        union = pred_area + target_area - intersect
        iou = self._safe_divide(intersect, union)
        dice = self._safe_divide(2 * intersect, pred_area + target_area)
        aupr = self._binned_aupr(pos_hist, total_hist)

        lesion_slice = slice(1, self.num_classes)
        metrics = {
            "mIoU": float(np.nanmean(iou[lesion_slice]) * 100),
            "mDice": float(np.nanmean(dice[lesion_slice]) * 100),
            "mAUPR": float(np.nanmean(aupr[lesion_slice]) * 100),
        }

        for offset, name in enumerate(LESION_NAMES, start=1):
            metrics[f"{name}_IoU"] = float(iou[offset] * 100)
            metrics[f"{name}_Dice"] = float(dice[offset] * 100)
            metrics[f"{name}_AUPR"] = float(aupr[offset] * 100)

        self._print_metric_table(metrics)
        return metrics

    def _print_metric_table(self, metrics):
        rows = [("mean", metrics["mIoU"], metrics["mDice"], metrics["mAUPR"])]
        for name in LESION_NAMES:
            rows.append((
                name,
                metrics[f"{name}_IoU"],
                metrics[f"{name}_Dice"],
                metrics[f"{name}_AUPR"],
            ))

        table = [
            "",
            "DDR lesion metrics:",
            "+----------------+----------+----------+----------+",
            "| lesion         | IoU      | Dice     | AUPR     |",
            "+----------------+----------+----------+----------+",
        ]
        for name, iou, dice, aupr in rows:
            table.append(
                f"| {name:<14} | {iou:>8.4f} | {dice:>8.4f} | {aupr:>8.4f} |"
            )
        table.append("+----------------+----------+----------+----------+")
        print_log("\n".join(table), logger="current")

    def _get_label_map(self, data_sample, name):
        value = self._get_pixel_data(data_sample, name)
        if value is None:
            raise ValueError(f"DDR evaluation requires {name}.")
        if value.ndim == 3 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 2:
            raise ValueError(f"{name} must have shape (H, W) or (1, H, W).")
        return value.detach().to(torch.long).cpu()

    def _extract_score(self, data_sample, target_shape):
        logits = self._get_pixel_data(data_sample, "seg_logits", default=None)
        if logits is None:
            raise ValueError("DDR AUPR requires seg_logits, not hard class predictions.")
        logits = logits.detach().float().cpu()
        expected_shape = (self.num_classes, *target_shape)
        if tuple(logits.shape) != expected_shape:
            raise ValueError(
                f"seg_logits shape {tuple(logits.shape)} does not match "
                f"the evaluation frame {expected_shape}.")
        return F.softmax(logits, dim=0)

    def _get_pixel_data(self, data_sample, name, default=None):
        if hasattr(data_sample, name):
            pixel_data = getattr(data_sample, name)
        elif isinstance(data_sample, dict) and name in data_sample:
            pixel_data = data_sample[name]
        else:
            return default

        if hasattr(pixel_data, "data"):
            return pixel_data.data
        if isinstance(pixel_data, dict) and "data" in pixel_data:
            return pixel_data["data"]
        return pixel_data

    def _safe_divide(self, numerator, denominator):
        out = np.full_like(numerator, np.nan, dtype=np.float64)
        valid = denominator > 0
        out[valid] = numerator[valid] / denominator[valid]
        return out

    def _binned_aupr(self, pos_hist, total_hist):
        aupr = np.full(self.num_classes, np.nan, dtype=np.float64)
        for class_idx in range(self.num_classes):
            positives = pos_hist[class_idx].sum()
            if positives <= 0:
                continue
            pos_desc = pos_hist[class_idx][::-1]
            total_desc = total_hist[class_idx][::-1]
            tp = np.cumsum(pos_desc)
            total = np.cumsum(total_desc)
            precision = np.divide(tp, total, out=np.zeros_like(tp), where=total > 0)
            aupr[class_idx] = np.sum(precision * pos_desc) / positives
        return aupr
