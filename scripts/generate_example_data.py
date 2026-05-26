from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dev_platform_constraints.examples import generate_sample_grid
from dev_platform_constraints.terrain_features import derive_terrain_features


def main() -> None:
    grid = generate_sample_grid()
    derive_terrain_features(grid)
    output_dir = ROOT / "data"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "sample_grid.npz"
    np.savez(output_path, **grid.layers)
    print(f"wrote {output_path}")
    print(f"layers: {', '.join(sorted(grid.layers))}")


if __name__ == "__main__":
    main()
