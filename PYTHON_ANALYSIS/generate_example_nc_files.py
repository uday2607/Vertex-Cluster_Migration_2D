#!/usr/bin/env python3
"""Generate small example .nc files for demos and tutorials.

Creates one SPV and one AVM example file per ensemble index, using the
real CellGPU filename convention (tokens separated by underscores).

Usage:
  python PYTHON_ANALYSIS/generate_example_nc_files.py --output-dir DATA/EXAMPLES --overwrite
"""

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PYTHON_ANALYSIS.example_data import create_example_nc, create_spv_example, create_avm_example


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate example CellGPU netCDF files")
    parser.add_argument("--output-dir", default="DATA/EXAMPLES",
                        help="Directory for generated .nc files")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing files")
    parser.add_argument("--ensembles", type=int, default=1,
                        help="Number of ensemble replicates to generate per scenario")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spv_dir = output_dir / "POS_GAMMA_HETERO"
    avm_dir = output_dir / "POS_GAMMA_VERTEX_HETERO"
    spv_dir.mkdir(parents=True, exist_ok=True)
    avm_dir.mkdir(parents=True, exist_ok=True)

    # Shared parameter set (matches the param script convention)
    shared_kw = dict(
        N=400, m_val=8, a_val=4,
        sa=3.7, sb=3.7,
        va=2, gamma=0.0,
        Ka=10, Kp=1,
        Dr_tp=0.005, vD_tp=0.02,
        IC=3,
        t_steps=200, dt=0.0001,
        overwrite=args.overwrite,
    )

    any_written = False

    for i in range(1, args.ensembles + 1):
        # SPV example
        spv_path = create_spv_example(spv_dir, ensem=i, **shared_kw)
        if spv_path.exists():
            print(f"[OK] SPV  {spv_path.name}")
            any_written = True
        else:
            print(f"[SKIP] SPV ensem {i} already exists (use --overwrite)")

        # AVM example
        avm_path = create_avm_example(avm_dir, ensem=i, **shared_kw)
        if avm_path.exists():
            print(f"[OK] AVM  {avm_path.name}")
            any_written = True
        else:
            print(f"[SKIP] AVM ensem {i} already exists (use --overwrite)")

    # Also create generic small test files (for backward compatibility with tests)
    generic_files = [
        output_dir / "example_spv_small_ensem_1.nc",
        output_dir / "example_spv_small_ensem_2.nc",
    ]
    for i, f in enumerate(generic_files, start=1):
        if f.exists() and not args.overwrite:
            print(f"[SKIP] {f.name}")
            continue
        create_example_nc(f, n_cells=6 + i, t_len=10 + i)
        print(f"[OK] generic {f.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
