"""One-time converter: stream data/data_2k_lw.json -> data/lcbench_cache.npz

Caches per-dataset matrices of validation accuracy (n_configs, 52) plus
final test accuracy vectors. Streaming via ijson (the JSON is ~1 GB).

Run: python experiments/convert_lcbench.py
"""

import sys
from pathlib import Path

import ijson
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "data_2k_lw_clean.json"  # Infinity literals stripped
OUT = ROOT / "data" / "lcbench_cache.npz"

TAG = "Train/val_accuracy"


def main():
    store: dict[str, np.ndarray] = {}
    cur_ds = None
    rows: dict[int, np.ndarray] = {}

    with open(SRC, "rb") as fh:
        # events: map_key at top level = dataset name; second level = config id
        parser = ijson.parse(fh)
        path_ds, path_cfg = None, None
        buf = []
        capture = False
        for prefix, event, value in parser:
            parts = prefix.split(".") if prefix else []
            if len(parts) == 0 and event == "map_key":
                if cur_ds is not None and rows:
                    n = max(rows) + 1
                    mat = np.full((n, 52), np.nan, dtype=np.float32)
                    for cid, arr in rows.items():
                        mat[cid, :len(arr)] = arr
                    store[cur_ds] = mat
                    print(f"{cur_ds}: {mat.shape}", flush=True)
                cur_ds = value
                rows = {}
            elif len(parts) == 1 and event == "map_key":
                path_cfg = int(value)
            elif prefix.endswith(f"log.{TAG}") and event == "start_array":
                capture, buf = True, []
            elif capture and event == "number":
                buf.append(float(value))
            elif capture and event == "end_array":
                capture = False
                rows[path_cfg] = np.array(buf, dtype=np.float32) / 100.0
        if cur_ds is not None and rows:
            n = max(rows) + 1
            mat = np.full((n, 52), np.nan, dtype=np.float32)
            for cid, arr in rows.items():
                mat[cid, :len(arr)] = arr
            store[cur_ds] = mat
            print(f"{cur_ds}: {mat.shape}", flush=True)

    np.savez_compressed(OUT, **store)
    print(f"saved {OUT} with {len(store)} datasets")


if __name__ == "__main__":
    main()
