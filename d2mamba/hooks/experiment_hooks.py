"""DDR experiment metric logging hooks."""

import math
import os
import zipfile
from copy import deepcopy
from datetime import datetime
from xml.sax.saxutils import escape

import torch
from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmengine.registry import HOOKS


def _is_main_process():
    try:
        from mmengine.dist import is_main_process
        return is_main_process()
    except Exception:
        return True


def _to_scalar(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().float().cpu().item()
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    return value


def _flatten_metrics(metrics):
    row = {}
    if not metrics:
        return row
    for key, value in metrics.items():
        value = _to_scalar(value)
        if value is not None:
            row[str(key)] = value
    return row


def _write_xlsx(path, rows, sheet_name="metrics"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    headers = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)

    def cell(col_idx, row_idx, value):
        col = ""
        n = col_idx + 1
        while n:
            n, rem = divmod(n - 1, 26)
            col = chr(65 + rem) + col
        ref = f"{col}{row_idx}"
        if value is None:
            return f'<c r="{ref}"/>'
        if isinstance(value, bool):
            return f'<c r="{ref}" t="b"><v>{int(value)}</v></c>'
        if isinstance(value, (int, float)):
            return f'<c r="{ref}"><v>{value}</v></c>'
        return f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'

    rows_xml = []
    rows_xml.append(
        '<row r="1">' + "".join(cell(i, 1, header) for i, header in enumerate(headers)) + "</row>"
    )
    for row_idx, row in enumerate(rows, start=2):
        rows_xml.append(
            f'<row r="{row_idx}">'
            + "".join(cell(i, row_idx, row.get(header)) for i, header in enumerate(headers))
            + "</row>"
        )

    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<sheetData>"
        + "".join(rows_xml)
        + "</sheetData></worksheet>"
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets><sheet name="{escape(sheet_name)}" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        "</Relationships>"
    )
    types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        "</Types>"
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", types_xml)
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)


@HOOKS.register_module()
class ValTestXlsxHook(Hook):
    """Run test after every validation and persist training/metric xlsx files."""

    priority = "LOW"

    def __init__(
        self,
        history_filename="train_history.xlsx",
        best_filename="best_metrics.xlsx",
        greater_keys=("IoU", "Dice", "AUPR", "Acc", "Fscore", "mDice", "mIoU", "mAUPR"),
        lower_keys=("loss",),
        flush_interval=50,
        run_test_after_val=False,
    ):
        self.history_filename = history_filename
        self.best_filename = best_filename
        self.greater_keys = tuple(greater_keys)
        self.lower_keys = tuple(lower_keys)
        self.history_rows = []
        self.best_rows = {}
        self._running_extra_test = False
        self.flush_interval = max(1, int(flush_interval))
        self.run_test_after_val = run_test_after_val

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        return

    def after_val_epoch(self, runner, metrics=None):
        if _is_main_process():
            self._record_eval(runner, "val", metrics)
        if not self.run_test_after_val or self._running_extra_test or not hasattr(runner, "test_loop"):
            return
        if getattr(runner, "test_loop", None) is None:
            return
        self._running_extra_test = True
        was_training = runner.model.training
        try:
            test_metrics = runner.test_loop.run()
        finally:
            if was_training:
                runner.model.train()
            self._running_extra_test = False
        if _is_main_process():
            self._record_eval(runner, "test", test_metrics)
            print_log("Finished paired test evaluation after validation.", logger="current")

    def after_test_epoch(self, runner, metrics=None):
        if _is_main_process() and not self._running_extra_test:
            self._record_eval(runner, "test", metrics)

    def after_train(self, runner):
        if _is_main_process():
            self._save(runner)

    def _record_eval(self, runner, phase, metrics):
        row = self._base_row(runner, phase)
        row.update(_flatten_metrics(metrics))
        self.history_rows.append(row)
        self._update_best(row)
        self._save(runner)

    def _base_row(self, runner, phase):
        return {
            "time": datetime.now().isoformat(timespec="seconds"),
            "iter": int(getattr(runner, "iter", 0)),
            "epoch": int(getattr(runner, "epoch", 0)),
            "phase": phase,
        }

    def _update_best(self, row):
        phase = row.get("phase")
        for key, value in row.items():
            if not isinstance(value, (int, float)) or key in {"iter", "epoch"}:
                continue
            mode = self._metric_mode(key)
            if mode is None:
                continue
            best_key = f"{phase}/{key}"
            old = self.best_rows.get(best_key)
            better = old is None or (mode == "greater" and value > old["value"]) or (
                mode == "lower" and value < old["value"]
            )
            if better:
                self.best_rows[best_key] = {
                    "metric": best_key,
                    "value": value,
                    "best_iter": row.get("iter"),
                    "best_epoch": row.get("epoch"),
                    "updated_time": row.get("time"),
                }

    def _metric_mode(self, key):
        lowered = key.lower()
        if any(token.lower() in lowered for token in self.lower_keys):
            return "lower"
        if any(token.lower() in lowered for token in self.greater_keys):
            return "greater"
        return None

    def _save(self, runner):
        work_dir = getattr(runner, "work_dir", ".")
        _write_xlsx(os.path.join(work_dir, self.history_filename), self.history_rows, "history")
        best_rows = [deepcopy(row) for _key, row in sorted(self.best_rows.items())]
        _write_xlsx(os.path.join(work_dir, self.best_filename), best_rows, "best_metrics")
