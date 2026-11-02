"""Evaluation configuration and checkpoint checks for DDR experiments."""

from copy import deepcopy

from mmengine.model import is_model_wrapper


def select_eval_split(cfg, split):
    if split == 'val':
        cfg.test_dataloader = deepcopy(cfg.val_dataloader)
        cfg.test_evaluator = deepcopy(cfg.val_evaluator)
    dataset = cfg.test_dataloader.dataset
    if dataset.type == 'DDRLesionDataset':
        stochastic = {'RandomResize', 'RandomCrop', 'RandomFlip',
                      'PhotoMetricDistortion', 'RandomRotate', 'RandomRotFlip'}
        found = [step['type'] for step in dataset.pipeline if step['type'] in stochastic]
        if found:
            raise ValueError(f'DDR {split} evaluation contains training augmentation: {found}')


def check_checkpoint_compatibility(model, checkpoint):
    if is_model_wrapper(model):
        model = model.module
    expected = model.state_dict()
    weights = checkpoint.get('state_dict', checkpoint)
    weights = {key.removeprefix('module.'): value for key, value in weights.items()}
    missing = sorted(expected.keys() - weights.keys())
    unexpected = sorted(weights.keys() - expected.keys())
    wrong_shape = {key: (tuple(weights[key].shape), tuple(expected[key].shape))
                   for key in expected.keys() & weights.keys()
                   if weights[key].shape != expected[key].shape}
    if missing or unexpected or wrong_shape:
        raise RuntimeError(f'Checkpoint mismatch: missing={missing}, unexpected={unexpected}, '
                           f'wrong_shape={wrong_shape}')
