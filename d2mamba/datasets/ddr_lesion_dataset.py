import csv
import os
import os.path as osp

from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class DDRLesionDataset(BaseSegDataset):
    """DDR lesion segmentation dataset backed by project CSV manifests."""

    METAINFO = dict(
        classes=(
            "background",
            "hard exudate",
            "microaneurysm",
            "soft exudate",
            "hemorrhage",
        ),
        palette=[
            [0, 0, 0],
            [220, 20, 60],
            [128, 0, 128],
            [255, 215, 0],
            [70, 130, 180],
        ],
    )

    def __init__(self, manifest_file=None, **kwargs):
        self.manifest_file = manifest_file
        super().__init__(
            img_suffix=kwargs.pop("img_suffix", ".jpg"),
            seg_map_suffix=kwargs.pop("seg_map_suffix", ".png"),
            reduce_zero_label=False,
            **kwargs,
        )

    def load_data_list(self):
        if self.manifest_file is None:
            return super().load_data_list()

        # Manifests ship with the repository (splits/); fall back to data_root.
        if osp.exists(self.manifest_file):
            manifest_path = osp.normpath(self.manifest_file)
        else:
            manifest_path = self._resolve_path(self.data_root, self.manifest_file)

        data_list = []

        with open(manifest_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                img_path = self._resolve_path(self.data_root, row["image"])
                seg_map = row.get("seg_map", "")
                data_info = dict(
                    img_path=img_path,
                    seg_map_path=self._resolve_path(self.data_root, seg_map) if seg_map else None,
                    label_map=self.label_map,
                    reduce_zero_label=self.reduce_zero_label,
                    seg_fields=[],
                    img_id=row.get("image_id", osp.splitext(osp.basename(img_path))[0]),
                )
                data_list.append(data_info)

        return data_list

    def _resolve_path(self, root, path):
        path = (path or "").strip()
        if not path:
            return ""
        path = path.replace("\\", os.sep)
        if osp.isabs(path):
            return osp.normpath(path)
        return osp.normpath(osp.join(root, path))
