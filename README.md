# Familial erythrocytosis multi-omics figure generation

This repository contains the Python code used to generate the main and supplementary SVG figures from the merged clinical, proteomic, and metabolomic dataset.

## Input

Place the merged CSV in the working directory, for example:

```text
Merged_by_Sample_ID_and_Group(4).csv
```

The scripts assume the same table structure used for the manuscript:

- `Sample_ID` and `Group` in the first columns
- clinical variables in the early block
- proteomics features in the middle block
- metabolomics features in the final block

Proteomics and metabolomics zero values are treated as missing, because they represent non-detections/below-detection values rather than true biological zeros.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Generate all figures

```bash
python generate_all_erythrocytosis_figures.py \
  --input "Merged_by_Sample_ID_and_Group(4).csv" \
  --outdir figures
```

The standardized manuscript outputs are:

```text
figures/Figure_1.svg
figures/Figure_2.svg
figures/Figure_3.svg
figures/Supplementary_Figure_1.svg
figures/Supplementary_Figure_2.svg
figures/Supplementary_Figure_3.svg
figures/Supplementary_Figure_4.svg
```

The scripts also preserve intermediate named versions used during figure development.

## Notes

- SVG text is kept editable by setting `matplotlib.rcParams['svg.fonttype'] = 'none'`.
- Main figure generation is handled by `scripts/generate_main_figures_final_realdata.py`.
- Supplementary figure generation is handled by `scripts/refine_supp_figs.py`, `scripts/remake_supp_fig3_fig4_final.py`, and `scripts/make_fig4_allgroup_perf_quick.py`.
- The code performs preprocessing, missing-value handling, autoscaling, PLS-DA, pathway scoring, differential feature selection, correlation/network analyses, random-forest classification, LOOCV, and NDD/NDD-PDE4 neighborhood mapping.
