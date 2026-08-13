# Red Blood Cell Multi-Omics in Familial Erythrocytosis

Reproducible analysis code and processed deidentified data accompanying the study:

**Red blood cell multi-omics delineates familial erythrocytosis endotypes and maps unresolved cases**

Domenico Roberti, Paolo Montaldo, Monika Dzieciatkowska, Daniel Stephenson, Debora Bencivenga, Saverio Scianguetta, Teresa Palma, Elisa Bonetti, Angela Maggio, Ilaria Fotzi, Giovanna D’Urzo, Silvia Fasoli, Irene D’Alba, Paola C. Corti, Ada Zaccaron, Anna Grandone, Immacolata Tartaglione, Maddalena Casale, Fulvio Della Ragione, Adriana Borriello, Silverio Perrotta, and Angelo D’Alessandro.

## Overview

Familial erythrocytosis comprises genetically and clinically heterogeneous disorders involving altered oxygen sensing, erythropoietic signaling, oxygen delivery, and red blood cell physiology.

This repository contains the processed deidentified data and Python code used for the revised statistical analyses and generation of the main and supplementary figures.

The analytical framework distinguishes:

1. descriptive sample-level molecular analyses;
2. participant-level clinical analyses;
3. participant-grouped supervised classification; and
4. exploratory molecular-neighborhood mapping of unresolved cases.

The molecular-neighborhood analyses are hypothesis-generating and are not intended to replace clinical evaluation, genetic testing, oxygen-affinity studies, or other established approaches to diagnosis.

## Repository structure

```text
Erythrocytosis/
├── README.md
├── requirements.txt
├── scripts/
│   ├── run_statistical_revision.py
│   ├── generate_main_figures_revised.py
│   └── generate_revised_supplementary_figures.py
└── working/
    ├── analysis_compatibility_full.csv
    └── analysis_metadata.csv
```

Generated statistical results and figure files are **not version-controlled**. They are produced directly from the supplied data and scripts when the analyses are run.

The primary statistical workflow creates:

```text
working/analysis_results/
```

and the figure-generation scripts can subsequently be used to regenerate the main and supplementary figures.

## Input data

### `analysis_compatibility_full.csv`

Processed clinical, red blood cell proteomic, and metabolomic data matrix used for the revised analyses.

The existing column structure is used by the analytical scripts to identify the clinical, proteomic, and metabolomic data blocks and should therefore be preserved.

### `analysis_metadata.csv`

Sample- and participant-level metadata used to:

- distinguish samples from independent participants;
- identify the primary omics cohort;
- specify one baseline observation per participant where appropriate;
- identify repeated samples;
- annotate group membership;
- provide EPO at diagnosis; and
- provide thrombosis-before-age-40 status.

## Cohort and unit of analysis

The reconciled dataset contains **65 samples from 63 participants**.

Two VHL-heterozygous Family-group samples, F2_1 and F6_1, were excluded from the primary omics workflow following the original unsupervised quality-control review. The primary omics cohort therefore contains **63 samples from 61 participants**.

The defined erythrocytosis classification cohort contains **40 samples from 39 participants** representing five etiologic classes:

- VHL
- EGLN1
- EPAS1
- EPOR
- high-affinity hemoglobin (HAH)

Unresolved erythrocytosis and PDE4-associated acrodysostosis with unresolved erythrocytosis were not used as defined classifier training classes.

P14/P14MED and P47/P47MED represent repeated samples from the same participants. All samples belonging to a participant are withheld together during supervised validation so that observations from the same participant cannot appear in both the training and test sets.

## Primary predictor specification

Age, sex, erythropoietin (EPO), and thrombosis were excluded from the primary clinical predictor matrix.

Age and sex were evaluated separately as potential confounders.

EPO at diagnosis was evaluated as a secondary clinical variable and in a clinical-plus-EPO sensitivity model.

Thrombosis before age 40 was analyzed as a participant-level clinical outcome.

## Participant-grouped classification

Supervised classification was restricted to the five defined erythrocytosis classes: VHL, EGLN1, EPAS1, EPOR, and HAH.

Validation used **leave-one-participant-out cross-validation**.

For every validation fold:

1. all samples belonging to one participant were withheld;
2. preprocessing was fitted using only the remaining participants;
3. zero-valued proteomic and metabolomic measurements were treated as non-detections;
4. omics features were retained when detected in at least 35% of the training observations;
5. retained positive-valued omics features were log2-transformed;
6. missing values were median-imputed;
7. variables were standardized;
8. ANOVA-based `SelectKBest` feature selection was performed within the training fold;
9. a balanced multinomial logistic-regression classifier was fitted; and
10. the held-out participant was predicted.

When the held-out participant had more than one sample, class probabilities were averaged across samples to yield a single participant-level prediction.

Logistic regression used:

```text
solver = lbfgs
class_weight = balanced
C = 1.0
max_iter = 5000
```

The following predictor sets were evaluated:

| Model | Features retained |
|---|---:|
| Age/sex only | 2 |
| Primary clinical | up to 12 |
| Clinical + age/sex | up to 14 |
| Clinical + EPO | up to 13 |
| Metabolome | 25 |
| Proteome | 25 |
| Combined clinical + proteome + metabolome | 35 |

Performance is summarized once per participant using:

- overall accuracy;
- exact Clopper-Pearson 95% confidence intervals;
- balanced accuracy;
- confusion matrices; and
- per-class recall.

## Mapping unresolved erythrocytosis

Unresolved participants were not included as training classes in the defined-subgroup classifiers.

Four classifiers were fitted to the complete defined erythrocytosis cohort using:

- primary clinical variables;
- metabolomics;
- proteomics; and
- combined clinical, metabolomic, and proteomic predictors.

Each model generated a probability distribution across the five defined molecular neighborhoods:

```text
VHL | EGLN1 | EPAS1 | EPOR | HAH
```

For participants represented by repeated samples, probabilities were first averaged across samples.

The probability vectors generated by the clinical, metabolomic, proteomic, and combined models were then averaged with equal weighting to produce a participant-level **consensus molecular-neighborhood probability vector**.

The class receiving the highest consensus probability is reported as the closest modeled molecular neighborhood.

These probabilities represent exploratory molecular similarity and **must not be interpreted as diagnostic assignments or causal gene calls**.

## Random-forest proximity analyses

Main Figure 3 also contains exploratory participant-level random-forest analyses.

For these analyses, standardized measurements were averaged to one observation per participant.

A random forest containing **600 trees** was fitted using balanced-subsample class weighting and square-root feature sampling. Pairwise random-forest proximity was defined as the fraction of trees in which two participants occupied the same terminal leaf.

A dissimilarity matrix was calculated as:

```text
distance = 1 - random-forest proximity
```

and projected into two dimensions using multidimensional scaling.

A separate binary random forest containing **800 trees** was used to contrast unresolved/PDE4-associated participants with participants belonging to the five defined erythrocytosis classes. Gini-based feature importance was used descriptively to identify variables contributing to this distinction.

These analyses are exploratory and are not independent validation tests.

## Additional statistical analyses

### Age and sex

Age was compared across the five defined groups using the Kruskal-Wallis test.

Sex distribution was evaluated using a chi-square statistic and **50,000 permutations** preserving group sizes.

An age/sex-only participant-grouped classifier was also evaluated to determine whether these demographic variables alone could distinguish etiologic groups.

### Erythropoietin

EPO at diagnosis was summarized by group and compared across the five defined etiologic groups using the Kruskal-Wallis test.

Pairwise comparisons were performed using two-sided Mann-Whitney U tests with Benjamini-Hochberg correction.

A clinical-plus-EPO classifier was evaluated as a secondary sensitivity analysis.

### Thrombosis

Thrombosis was defined as a reported thrombotic event before 40 years of age.

VHL-associated erythrocytosis was compared with all other affected participants using Fisher's exact test.

## Relationship between analyses and figures

### Main Figure 1

Cohort structure, supervised PLS-DA visualization, and clinical, metabolomic, and proteomic heat maps.

PLS-DA is used as a descriptive supervised dimension-reduction method and is not interpreted as externally validated diagnostic performance.

### Main Figure 2

Pathway-level metabolomic and proteomic remodeling together with exploratory clinical-omics Spearman correlation analyses.

### Main Figure 3

Panels A-D show participant-grouped classification of the five defined erythrocytosis groups using clinical, metabolomic, proteomic, and combined predictor sets.

Panel E shows participant-level consensus molecular-neighborhood probabilities for unresolved erythrocytosis.

Subsequent panels provide exploratory random-forest proximity and unresolved-versus-defined feature analyses.

### Supplementary Figure 1

Cohort accounting and unsupervised quality-control analyses, including missingness, data-layer completeness, sample correlations, and PCA.

### Supplementary Figure 2

Exploratory subgroup-versus-remainder metabolomic and proteomic comparisons using Mann-Whitney U testing with Benjamini-Hochberg false-discovery-rate correction.

### Supplementary Figure 3

Participant-level clinical context and representative clinical, metabolomic, and proteomic distributions, including EPO and thrombosis.

### Supplementary Figure 4

Expanded participant-grouped classifier validation and predictor-set sensitivity analyses.

### Supplementary Figure 5

Conceptual framework illustrating how molecular-neighborhood information may be integrated with phenotype, biological pathways, and genomic information to prioritize future genomic reanalysis and functional studies.

The candidate-prioritization weights displayed in Supplementary Figure 5 are illustrative. **No composite candidate-gene prioritization score was trained, optimized, or validated using the present cohort.**

## Installation

Python 3 is required.

Install the required packages from the repository root:

```bash
pip install -r requirements.txt
```

The minimum dependencies are:

```text
numpy
pandas
scipy
scikit-learn
matplotlib
```

## Reproducing the statistical analyses

From the repository root, run:

```bash
python scripts/run_statistical_revision.py
```

The script creates:

```text
working/analysis_results/
```

and generates files including:

```text
classifier_metrics.csv
participant_predictions_*.csv
confusion_*.csv
unknown_probabilities_by_feature_set.csv
unknown_consensus_probabilities.csv
epo_summary.csv
epo_pairwise_defined_groups.csv
thrombosis_summary.csv
analysis_summary.json
```

It also generates statistical summary graphics used in the revised supplementary analyses.

The generated results are intentionally not stored in the repository because they can be reproduced directly from the supplied data and code.

## Reproducing the main figures

After running the statistical analysis, the main-figure script uses the generated files in `working/analysis_results/`.

For example:

```bash
export FIG_DATA="working/analysis_compatibility_full.csv"
export FIG_META="working/analysis_metadata.csv"
export FIG_ANALYSIS_RESULTS="working/analysis_results"
export FIG_OUTDIR="main_figures"

python scripts/generate_main_figures_revised.py
```

On Windows PowerShell, the corresponding environment variables can be set with:

```powershell
$env:FIG_DATA="working/analysis_compatibility_full.csv"
$env:FIG_META="working/analysis_metadata.csv"
$env:FIG_ANALYSIS_RESULTS="working/analysis_results"
$env:FIG_OUTDIR="main_figures"

python scripts/generate_main_figures_revised.py
```

The output directory is created locally and does not need to be committed to the repository.

## Reproducing the supplementary figures

After running the statistical analysis:

```bash
python scripts/generate_revised_supplementary_figures.py \
    --data working/analysis_compatibility_full.csv \
    --metadata working/analysis_metadata.csv \
    --results working/analysis_results \
    --outdir supplementary_figures
```

The generated SVG and PNG files are written to the requested output directory and are not stored in the repository.

**Note:** this command assumes the unused legacy `--merged` argument has been removed from `generate_revised_supplementary_figures.py`.

## Reproducibility safeguards

The revised workflow incorporates several safeguards intended to avoid information leakage and inappropriate inflation of classifier performance:

- samples are explicitly linked to participants;
- repeated samples from a participant are withheld together;
- classifier performance is summarized once per participant;
- omics detection filtering is performed within each training fold;
- imputation and scaling are refitted within each training fold;
- supervised feature selection is performed within each training fold;
- age, sex, EPO, and thrombosis are excluded from the primary predictor matrix;
- unresolved participants are excluded from defined-class classifier training; and
- unresolved-case probabilities are interpreted as molecular-neighborhood similarities rather than diagnostic assignments.

## Generated files

Statistical results and figures are intentionally not committed to this repository.

They can be recreated from the provided data and scripts, allowing the repository to serve as a compact reproducible analysis package rather than an archive of derivative files.

Users wishing to retain their locally generated outputs may add the following directories to `.gitignore`:

```text
working/analysis_results/
main_figures/
supplementary_figures/
```

## Data and code availability

The repository provides the deidentified processed data and analysis code required to reproduce the statistical analyses and manuscript figures.

The manuscript and accompanying supplementary material provide the biological interpretation, experimental methods, and additional methodological details.

## Contact

**Domenico Roberti, MD/PhD**  
University of Campania “Luigi Vanvitelli”  
domenico.roberti@unicampania.it

**Angelo D’Alessandro, PhD**  
University of Colorado Anschutz Medical Campus  
angelo.dalessandro@cuanschutz.edu