#!/usr/bin/env python3
"""
Generate all main and supplementary SVG figures for the familial erythrocytosis
multi-omics manuscript from the merged real-data table.

Expected input:
    Merged_by_Sample_ID_and_Group(4).csv

Usage:
    python generate_all_erythrocytosis_figures.py \
        --input "Merged_by_Sample_ID_and_Group(4).csv" \
        --outdir figures

The script wraps the validated figure-generation scripts in ./scripts and writes
publication-ready SVG files with editable text (matplotlib svg.fonttype='none').
"""

from __future__ import annotations

import argparse
import os
import runpy
import shutil
from pathlib import Path


def run_script(script_path: Path, data_path: Path, outdir: Path) -> None:
    """Run one of the figure scripts with shared input/output environment."""
    os.environ["FIG_DATA"] = str(data_path.resolve())
    os.environ["FIG_OUTDIR"] = str(outdir.resolve())
    print(f"\n[erythrocytosis-figures] Running {script_path.name}")
    runpy.run_path(str(script_path.resolve()), run_name="__main__")


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  wrote {dst.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all erythrocytosis main and supplementary figures as SVG.")
    parser.add_argument("--input", required=True, help="Path to merged CSV data table.")
    parser.add_argument("--outdir", default="figures", help="Output directory for SVG figures and logs.")
    args = parser.parse_args()

    data_path = Path(args.input)
    outdir = Path(args.outdir)
    script_dir = Path(__file__).resolve().parent / "scripts"
    outdir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {data_path}")

    # 1. Main Figures 1-3.
    run_script(script_dir / "generate_main_figures_final_realdata.py", data_path, outdir)

    # 2. Supplementary Figures 1-4 refined base versions.
    run_script(script_dir / "refine_supp_figs.py", data_path, outdir)

    # 3. Supplementary Figure 3 final pass and Supplementary Figure 4 intermediate pass.
    run_script(script_dir / "remake_supp_fig3_fig4_final.py", data_path, outdir)

    # 4. Supplementary Figure 4 final pass with all-group classifier performance.
    run_script(script_dir / "make_fig4_allgroup_perf_quick.py", data_path, outdir)

    # Standardized output names for manuscript submission.
    standard = {
        "Main_Figure_1_final_realdata.svg": "Figure_1.svg",
        "Main_Figure_2_final_realdata.svg": "Figure_2.svg",
        "Main_Figure_3_final_realdata.svg": "Figure_3.svg",
        "Supplementary_Figure_1_refined.svg": "Supplementary_Figure_1.svg",
        "Supplementary_Figure_2_refined.svg": "Supplementary_Figure_2.svg",
        "Supplementary_Figure_3_final.svg": "Supplementary_Figure_3.svg",
        "Supplementary_Figure_4_final_all_group_performance.svg": "Supplementary_Figure_4.svg",
    }
    print("\n[erythrocytosis-figures] Standardizing output filenames")
    for src_name, dst_name in standard.items():
        copy_if_exists(outdir / src_name, outdir / dst_name)

    print("\nDone. Primary SVG outputs:")
    for name in standard.values():
        p = outdir / name
        if p.exists():
            print(f"  {p}")


if __name__ == "__main__":
    main()
