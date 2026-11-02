"""Ignore-safe cross entropy for DDR segmentation."""

import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS


@MODELS.register_module()
class SafeCrossEntropyLoss(nn.Module):
    """Cross entropy that ignores 255 before applying class weights."""

    def __init__(
        self,
        use_sigmoid=False,
        loss_weight=1.0,
        class_weight=None,
        ignore_index=255,
        loss_name="loss_ce",
    ):
        super().__init__()
        if use_sigmoid:
            raise NotImplementedError("SafeCrossEntropyLoss only supports softmax CE.")
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self, pred, target, weight=None, ignore_index=None, avg_factor=None, reduction_override=None):
        ignore_index = self.ignore_index if ignore_index is None else ignore_index
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)
        target = target.long()

        num_classes = pred.size(1)
        valid = (target != ignore_index) & (target >= 0) & (target < num_classes)
        if not valid.any():
            return pred.sum() * 0.0

        class_weight = None
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)

        pred = pred.permute(0, 2, 3, 1)[valid]
        target = target[valid]
        return F.cross_entropy(pred, target, weight=class_weight, reduction="mean") * self.loss_weight

    @property
    def loss_name(self):
        return self._loss_name
