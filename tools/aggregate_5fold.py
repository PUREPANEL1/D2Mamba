"""Aggregate held-out DDR validation metrics across five folds."""

import argparse
import os
import statistics
import zipfile
import xml.etree.ElementTree as ET
from copy import deepcopy


NS = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _cell_value(cell):
    inline = cell.find("main:is/main:t", NS)
    if inline is not None:
        return inline.text or ""
    value = cell.find("main:v", NS)
    if value is None or value.text is None:
        return ""
    text = value.text
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return int(number)
    return number


def _read_xlsx_rows(path):
    with zipfile.ZipFile(path) as zf:
        sheet_xml = zf.read("xl/worksheets/sheet1.xml")
    root = ET.fromstring(sheet_xml)
    rows = []
    for row_node in root.findall(".//main:row", NS):
        rows.append([_cell_value(cell) for cell in row_node.findall("main:c", NS)])
    if not rows:
        return []
    headers = [str(header) for header in rows[0]]
    parsed = []
    for values in rows[1:]:
        parsed.append({headers[i]: values[i] if i < len(values) else "" for i in range(len(headers))})
    return parsed


def _write_xlsx(path, rows, sheet_name="fivefold_val"):
    from xml.sax.saxutils import escape

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
        if value == "" or value is None:
            return f'<c r="{ref}"/>'
        if isinstance(value, bool):
            return f'<c r="{ref}" t="b"><v>{int(value)}</v></c>'
        if isinstance(value, (int, float)):
            return f'<c r="{ref}"><v>{value}</v></c>'
        return f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'

    rows_xml = [
        '<row r="1">' + "".join(cell(i, 1, header) for i, header in enumerate(headers)) + "</row>"
    ]
    for row_idx, row in enumerate(rows, start=2):
        rows_xml.append(
            f'<row r="{row_idx}">'
            + "".join(cell(i, row_idx, row.get(header, "")) for i, header in enumerate(headers))
            + "</row>"
        )

    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<sheetData>" + "".join(rows_xml) + "</sheetData></worksheet>"
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


def _best_eval_row(work_dir, metric, phase):
    history_path = os.path.join(work_dir, "train_history.xlsx")
    rows = _read_xlsx_rows(history_path)
    eval_rows = [
        row for row in rows
        if row.get("phase") == phase and isinstance(row.get(metric), (int, float))
    ]
    if not eval_rows:
        raise RuntimeError(f"No {phase} rows with metric {metric} in {history_path}")
    return max(eval_rows, key=lambda row: row[metric])


def _numeric_metric_keys(rows):
    blocked = {"fold", "stat", "iter", "epoch", "best_by"}
    keys = []
    for row in rows:
        for key, value in row.items():
            if key not in blocked and isinstance(value, (int, float)) and key not in keys:
                keys.append(key)
    return keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("val", "test"), default="val")
    parser.add_argument("--metric", default="mDice", help="Val metric used to pick each fold best row.")
    parser.add_argument("--output", default=None, help="Default: work_dirs/fivefold_<phase>_summary.xlsx")
    parser.add_argument(
        "--fold-work-dirs",
        nargs="*",
        default=[f"work_dirs/d2mamba_ddr_fold{i}" for i in range(5)],
    )
    args = parser.parse_args()

    fold_rows = []
    for fold_idx, work_dir in enumerate(args.fold_work_dirs):
        row = deepcopy(_best_eval_row(work_dir, args.metric, args.phase))
        row["fold"] = fold_idx
        row["stat"] = "fold"
        row["best_by"] = args.metric
        fold_rows.append(row)

    metric_keys = _numeric_metric_keys(fold_rows)
    mean_row = {"fold": "", "stat": "mean", "best_by": args.metric}
    var_row = {"fold": "", "stat": "variance", "best_by": args.metric}
    for key in metric_keys:
        values = [row[key] for row in fold_rows if isinstance(row.get(key), (int, float))]
        if values:
            mean_row[key] = statistics.mean(values)
            var_row[key] = statistics.pvariance(values)

    out_rows = fold_rows + [mean_row, var_row]
    if args.output is None:
        args.output = f"work_dirs/fivefold_{args.phase}_summary.xlsx"
    _write_xlsx(args.output, out_rows)

    display_keys = [key for key in ("mIoU", "mDice", "mAUPR", "EX_IoU", "MA_IoU", "SE_IoU", "HE_IoU") if key in metric_keys]
    print(f"Selected best {args.phase} row in each fold by {args.metric}")
    print("fold\titer\t" + "\t".join(display_keys))
    for row in fold_rows:
        print(
            f"{row['fold']}\t{row.get('iter', '')}\t"
            + "\t".join(f"{row.get(key, ''):.4f}" if isinstance(row.get(key), float) else str(row.get(key, "")) for key in display_keys)
        )
    print(
        "mean\t\t"
        + "\t".join(f"{mean_row.get(key, ''):.4f}" if isinstance(mean_row.get(key), float) else str(mean_row.get(key, "")) for key in display_keys)
    )
    print(
        "variance\t\t"
        + "\t".join(f"{var_row.get(key, ''):.4f}" if isinstance(var_row.get(key), float) else str(var_row.get(key, "")) for key in display_keys)
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
