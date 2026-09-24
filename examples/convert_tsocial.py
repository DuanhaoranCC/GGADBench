"""Convert the optional T-Social DGL graph to the benchmark NPZ format."""

import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("GAD_DATA_ROOT") or PROJECT_ROOT / "Dataset").expanduser().resolve()
REAL_DIR = DATA_ROOT / "real"
SRC = REAL_DIR / "tsocial"
OUT = REAL_DIR / "tsocial.npz"


def main():
    if not SRC.exists():
        sys.exit(f"File not found: {SRC}")
    import dgl

    print(f"dgl {dgl.__version__}: loading {SRC} ...")
    g = dgl.load_graphs(str(SRC))[0][0]
    n = g.num_nodes()
    u, v = g.edges()
    ei = np.stack([u.numpy().astype(np.int32), v.numpy().astype(np.int32)], axis=1)  # (E,2)
    x = g.ndata["feature"].numpy()  # (N,10) int
    y = g.ndata["label"].numpy().astype(np.int8)  # 0/1
    print(f"  n={n} edges={ei.shape[0]} featdim={x.shape[1]} anomaly={int((y==1).sum())}")
    np.savez(OUT, edge_index=ei, x=x, y=y)
    print(f"saved -> {OUT}  ({OUT.stat().st_size/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
