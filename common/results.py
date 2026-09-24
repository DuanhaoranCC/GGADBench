"""CSV result merging and single-source versus multi-source summaries."""

import csv
from evaluation import build_metric_delta


def _merge(path, new_rows):
    """Merge new result rows, replacing existing rows with matching experiment keys."""

    def key(r):
        return (r["method"], r["src_type"], r["tgt_type"], r["mode"], r["target"])

    new_keys = {key(r) for r in new_rows}
    old = []
    if path.exists():
        for r in csv.DictReader(open(path, encoding="utf-8")):
            r.setdefault("src_type", "real")
            r.setdefault("tgt_type", "real")
            if key(r) not in new_keys:
                old.append(r)
    return old + new_rows


def _build_delta(detail):
    return build_metric_delta(detail)


def _write(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
