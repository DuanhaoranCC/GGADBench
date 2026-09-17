"""Generate two small synthetic graphs for an offline installation smoke test.

The files use the benchmark's default source/target names for compatibility.
They are artificial data and must never be used as reported benchmark results.
"""

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp


def create_demo_data(output):
    """Write deterministic demo graphs without replacing existing dataset files."""
    output = Path(output).expanduser().resolve()
    real = output / "real"
    real.mkdir(parents=True, exist_ok=True)
    (output / "fake").mkdir(exist_ok=True)
    for name, seed in (("Facebook", 7), ("Disney", 11)):
        path = real / f"{name}.mat"
        if path.exists():
            raise FileExistsError(f"Refusing to replace existing data: {path}")
        rng = np.random.RandomState(seed)
        count, dim = 128, 16
        labels = np.zeros(count, dtype=np.int64)
        labels[-24:] = 1
        features = rng.normal(0.5, 0.1, size=(count, dim))
        features[labels == 1] += rng.normal(0.0, 0.6, size=(24, dim))
        edges = np.triu(rng.uniform(size=(count, count)) < 0.06, k=1)
        # A ring guarantees that every node has at least two neighbors.
        indices = np.arange(count)
        edges[indices, (indices + 1) % count] = True
        adjacency = sp.csr_matrix((edges | edges.T).astype(np.float32))
        sio.savemat(
            path,
            {
                "Network": adjacency,
                "Attributes": sp.csr_matrix(features.astype(np.float32)),
                "Label": labels[:, None],
            },
        )
    (output / "README.txt").write_text(
        "SYNTHETIC SMOKE-TEST DATA ONLY. These files are not the real Facebook "
        "or Disney datasets. Do not report their scores as benchmark results.\n",
        encoding="utf-8",
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent / "demo_data"
    )
    args = parser.parse_args()
    output = create_demo_data(args.output)
    print(f"Synthetic demo data created in {output}")
    print("Run from the repository root:")
    print(
        "python run.py --suite ggad --methods neighbordiv --src-type real "
        "--tgt-type real --source-mode single --targets Disney --quick "
        f'--device cpu --data-root "{output}" --output-dir results/demo'
    )


if __name__ == "__main__":
    main()
