"""Run from the repository root: python -m unittest discover -s tests -v."""

import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import mmseg
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from d2mamba.transforms.retina_preprocess import RetinaPreprocess, _gray_crop_bounds
from d2mamba.metrics.ddr_metrics import DDRLesionMetric
from d2mamba.losses.safe_ce_loss import SafeCrossEntropyLoss
from d2mamba.evaluation.eval_checks import check_checkpoint_compatibility, select_eval_split


def load_postprocess():
    # Execute the installed MMSeg method without importing unrelated CUDA ops.
    model_root = Path(mmseg.__file__).parent / "models"
    wrapper_spec = importlib.util.spec_from_file_location(
        "ddr_test_resize", model_root / "utils" / "wrappers.py")
    wrappers = importlib.util.module_from_spec(wrapper_spec)
    wrapper_spec.loader.exec_module(wrappers)
    path = model_root / "segmentors" / "base.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "BaseSegmentor")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "postprocess_result")
    module = ast.parse("from __future__ import annotations")
    module.body.append(method)
    namespace = dict(PixelData=PixelData, SegDataSample=SegDataSample,
                     resize=wrappers.resize)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["postprocess_result"]


POSTPROCESS = load_postprocess()


def make_sample(target, pred=None, logits=None, **meta):
    sample = SegDataSample(metainfo=meta)
    sample.gt_sem_seg = PixelData(data=target[None])
    if pred is not None:
        sample.pred_sem_seg = PixelData(data=pred[None])
    if logits is not None:
        sample.seg_logits = PixelData(data=logits)
    return sample


def perfect_logits(target):
    return F.one_hot(target.long(), 5).permute(2, 0, 1).float() * 20 - 10


def evaluate(sample):
    metric = DDRLesionMetric(prefix="")
    metric.process(None, [sample])
    return metric.compute_metrics(metric.results)


class PreprocessTests(unittest.TestCase):
    def test_image_mask_and_evaluation_frame_match(self):
        img = np.zeros((23, 37, 3), dtype=np.uint8)
        img[2:21, 3:34] = 60
        mask = np.zeros((23, 37), dtype=np.uint8)
        mask[5:9, 8:13] = 2
        result = RetinaPreprocess(output_size=(16, 12))(
            dict(img=img, gt_seg_map=mask, seg_fields=["gt_seg_map"],
                 ori_shape=img.shape[:2]))
        self.assertEqual(result["img_shape"], (12, 16))
        self.assertEqual(result["ori_shape"], result["gt_seg_map"].shape)
        self.assertEqual(result["source_ori_shape"], (23, 37))
        self.assertEqual(result["retina_crop_bbox"], (2, 21, 3, 34))
        self.assertEqual(set(np.unique(result["gt_seg_map"])), {0, 2})

    def test_grayscale_crop_uses_loader_bgr_order(self):
        img = np.zeros((3, 3, 3), dtype=np.uint8)
        img[1, 1] = [0, 0, 30]
        self.assertEqual(_gray_crop_bounds(img), (1, 2, 1, 2))

    def test_perfect_prediction_survives_mmseg_postprocess(self):
        mask = np.zeros((23, 37), dtype=np.uint8)
        for label in range(1, 5):
            mask[2 + label * 3:4 + label * 3, 2 + label * 5:5 + label * 5] = label
        result = RetinaPreprocess(output_size=(16, 16))(
            dict(img=np.full((23, 37, 3), 60, dtype=np.uint8),
                 gt_seg_map=mask, seg_fields=["gt_seg_map"], ori_shape=(23, 37)))
        target = torch.from_numpy(result["gt_seg_map"]).long()
        sample = make_sample(target, ori_shape=result["ori_shape"])
        prediction = POSTPROCESS(SimpleNamespace(align_corners=False),
                                 perfect_logits(target)[None], [sample])[0]
        self.assertEqual(prediction.pred_sem_seg.data.shape[-2:], target.shape)
        metrics = evaluate(prediction)
        self.assertAlmostEqual(metrics["mDice"], 100.0)
        self.assertAlmostEqual(metrics["mAUPR"], 100.0)


class MetricTests(unittest.TestCase):
    def test_known_counts_and_ignored_pixels(self):
        target = torch.tensor([[0, 1, 1, 2, 3, 4, 255]])
        pred = torch.tensor([[1, 1, 0, 2, 3, 4, 1]])
        metrics = evaluate(make_sample(target, pred, perfect_logits(pred)))
        self.assertAlmostEqual(metrics["EX_Dice"], 50.0)
        self.assertAlmostEqual(metrics["EX_IoU"], 100 / 3)
        self.assertAlmostEqual(metrics["mDice"], 87.5)
        self.assertAlmostEqual(metrics["EX_AUPR"], 100 * (1 / 3 + 2 / 4) / 2)

    def test_evaluator_dict_representation(self):
        target = torch.tensor([[0, 1, 2], [3, 4, 0]])
        sample = make_sample(target, target, perfect_logits(target))
        self.assertAlmostEqual(evaluate(sample.to_dict())["mDice"], 100.0)

    def test_mismatched_prediction_is_rejected(self):
        target = torch.ones((4, 4), dtype=torch.long)
        pred = torch.ones((8, 8), dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "shape"):
            evaluate(make_sample(target, pred, perfect_logits(pred)))

    def test_mismatched_logits_are_rejected(self):
        target = torch.ones((4, 4), dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "shape"):
            evaluate(make_sample(target, target, torch.zeros(5, 8, 8)))

    def test_missing_logits_are_rejected(self):
        target = torch.ones((4, 4), dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "seg_logits"):
            evaluate(make_sample(target, target))

    def test_training_flip_metadata_is_rejected(self):
        target = torch.zeros((8, 8), dtype=torch.long)
        target[1:3, 1:3] = 1
        flipped = target.flip(-1)
        sample = make_sample(flipped, ori_shape=(8, 8), flip=True,
                             flip_direction="horizontal")
        prediction = POSTPROCESS(SimpleNamespace(align_corners=False),
                                 perfect_logits(flipped)[None], [sample])[0]
        self.assertFalse(torch.equal(prediction.pred_sem_seg.data[0], flipped))
        with self.assertRaisesRegex(ValueError, "flip"):
            evaluate(prediction)


class LossTests(unittest.TestCase):
    def test_all_ignored_targets_allow_backward(self):
        loss = SafeCrossEntropyLoss(class_weight=[0.5, 0.7, 1.5, 0.7, 1.1])
        pred = torch.randn(1, 5, 2, 2, requires_grad=True)
        value = loss(pred, torch.full((1, 2, 2), 255))
        value.backward()
        self.assertEqual(value.item(), 0.0)
        self.assertTrue(torch.equal(pred.grad, torch.zeros_like(pred)))


class ConfigTests(unittest.TestCase):
    def test_custom_imports_resolve_inside_d2mamba_package(self):
        for path in sorted((ROOT / "configs" / "d2mamba").glob("d2mamba_ddr_fold*.py")):
            cfg = Config.fromfile(path, import_custom_modules=False)
            for name in cfg.custom_imports.imports:
                with self.subTest(config=path.name, module=name):
                    self.assertTrue(name.startswith("d2mamba."))
                    spec = importlib.util.find_spec(name)
                    self.assertIsNotNone(spec)
                    self.assertTrue(Path(spec.origin).is_relative_to(ROOT))

    def test_val_selection_does_not_modify_val_dataset(self):
        cfg = Config.fromfile(ROOT / "configs/d2mamba/d2mamba_ddr_fold1.py",
                              import_custom_modules=False)
        select_eval_split(cfg, "val")
        self.assertEqual(cfg.test_dataloader.dataset.manifest_file, "splits/fold1_val.csv")
        cfg.test_dataloader.dataset.pipeline.append(dict(type="RandomFlip"))
        self.assertEqual(len(cfg.val_dataloader.dataset.pipeline), 4)
        with self.assertRaisesRegex(ValueError, "training augmentation"):
            select_eval_split(cfg, "test")

    def test_checkpoint_mismatch_is_rejected(self):
        model = torch.nn.Linear(2, 5)
        good = model.state_dict()
        check_checkpoint_compatibility(model, {"state_dict": good})
        check_checkpoint_compatibility(model, {"state_dict": {
            "module." + key: value for key, value in good.items()}})
        with self.assertRaisesRegex(RuntimeError, "missing"):
            check_checkpoint_compatibility(model, {"state_dict": {"weight": good["weight"]}})
        with self.assertRaisesRegex(RuntimeError, "wrong_shape"):
            check_checkpoint_compatibility(model, {"state_dict": dict(good, weight=torch.zeros(3, 2))})

    def test_all_folds_use_deterministic_identical_evaluation(self):
        for fold in range(5):
            with self.subTest(fold=fold):
                cfg = Config.fromfile(
                    ROOT / f"configs/d2mamba/d2mamba_ddr_fold{fold}.py",
                    import_custom_modules=False)
                val = cfg.val_dataloader.dataset.pipeline
                test = cfg.test_dataloader.dataset.pipeline
                self.assertEqual(val, test)
                self.assertEqual([step.type for step in test],
                                 ["LoadImageFromFile", "LoadAnnotations",
                                  "RetinaPreprocess", "PackSegInputs"])
                self.assertIn("RandomCrop", [step.type for step in
                                              cfg.train_dataloader.dataset.pipeline])
                self.assertEqual(cfg.model.decode_head.num_classes, 5)
                self.assertFalse(cfg.model.data_preprocessor.get("rgb_to_bgr", False))


if __name__ == "__main__":
    unittest.main()
