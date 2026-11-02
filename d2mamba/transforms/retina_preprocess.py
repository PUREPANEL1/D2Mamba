"""Deterministic retinal preprocessing for DDR segmentation."""

import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


def _gray_crop_bounds(img, tol=7):
    if img.ndim == 2:
        gray_img = img
    else:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray_img > tol
    if not mask.any():
        return None
    rows = np.where(mask.any(1))[0]
    cols = np.where(mask.any(0))[0]
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1


@TRANSFORMS.register_module()
class RetinaPreprocess(BaseTransform):
    """Preprocess BGR fundus images and evaluate in the resized retina frame.

    Image and annotation are cropped/resized together. The resulting frame is
    the reference for later augmentations and MMSeg prediction postprocessing;
    original file coordinates are retained separately for inspection.
    """

    def __init__(self, output_size=(1280, 1280), sigma_x=15, crop_tol=7, num_classes=5, ignore_index=255):
        self.output_size = tuple(output_size)
        self.sigma_x = sigma_x
        self.crop_tol = crop_tol
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def transform(self, results):
        img = results["img"]
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)

        results["source_ori_shape"] = img.shape[:2]
        bounds = _gray_crop_bounds(img, self.crop_tol)
        results["retina_crop_bbox"] = (
            tuple(int(value) for value in bounds) if bounds is not None
            else (0, img.shape[0], 0, img.shape[1])
        )
        if bounds is not None:
            y1, y2, x1, x2 = bounds
            img = img[y1:y2, x1:x2, :]
            for key in results.get("seg_fields", []):
                results[key] = results[key][y1:y2, x1:x2]
        width, height = self.output_size
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        for key in results.get("seg_fields", []):
            results[key] = cv2.resize(
                results[key],
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            invalid = (results[key] != self.ignore_index) & (
                (results[key] < 0) | (results[key] >= self.num_classes)
            )
            results[key][invalid] = self.ignore_index
        img = cv2.addWeighted(
            img,
            4,
            cv2.GaussianBlur(img, (0, 0), self.sigma_x),
            -4,
            128,
        )
        results["img"] = img.astype(np.uint8)
        results["img_shape"] = img.shape[:2]
        # GT now lives in this frame; restoring file dimensions would resample
        # predictions a second time and erode small lesions in the metric.
        results["ori_shape"] = img.shape[:2]
        results["scale_factor"] = (1.0, 1.0)
        return results
