# qDGT-NTA

Reproduction code for *Machine Learning-Enhanced DGT Passive Sampling Coupled with Non-Targeted Analysis for High-Throughput Monitoring of Chemical Pollutants in Aquatic Systems*. The repository contains the authors' pretrained XGBoost models, training feature arrays for applicability domain (AD) checks, example inputs, and the supplied supplementary workbook.

## Run with Docker

```bash
docker build -t qdgt-nta .
mkdir -p figures output
docker run --rm --user "$(id -u):$(id -g)" -e MPLCONFIGDIR=/tmp/matplotlib -v "$PWD/figures:/app/figures" qdgt-nta
```

The default command recreates data-supported figures from `supplementary file.xlsx`. To predict the example inputs:

```bash
docker run --rm --user "$(id -u):$(id -g)" -e MPLCONFIGDIR=/tmp/matplotlib -v "$PWD/output:/app/output" qdgt-nta predict --model d --input '/app/examples/data example-for-D.xlsx' --output /app/output/d.xlsx --ad
docker run --rm --user "$(id -u):$(id -g)" -e MPLCONFIGDIR=/tmp/matplotlib -v "$PWD/output:/app/output" qdgt-nta predict --model pos --input '/app/examples/data example-for-IE(+).xlsx' --output /app/output/pos.xlsx --ad
docker run --rm --user "$(id -u):$(id -g)" -e MPLCONFIGDIR=/tmp/matplotlib -v "$PWD/output:/app/output" qdgt-nta predict --model neg --input '/app/examples/data example-for-IE(-).xlsx' --output /app/output/neg.xlsx --ad
```

For your own CSV or XLSX file, mount its directory into the container and pass the mounted path. Inputs need `SMILES`; IE inputs also need numeric instrument and mobile phase columns in the order shown by the matching example. Invalid SMILES remain in the output with empty prediction and AD fields. AD thresholds may be overridden with `--similarity` and `--count`.

## Local installation

Use Python 3.10, then `pip install -e .`. Run `qdgt-nta figures --output figures` or `qdgt-nta predict --model d --input examples/data\ example-for-D.xlsx --output output/d.xlsx --ad`.

## Guided notebook

[Experiments tour](notebooks/experiments_tour.ipynb) walks through the supplementary datasets, feature layouts, example predictions, AD thresholds, validation results, pollutant screening, and reproduced figures. Run it from the repository root:

```bash
python3.10 -m pip install -e ".[notebook]"
jupyter lab notebooks/experiments_tour.ipynb
```

Or launch Jupyter from the Docker image after building it:

```bash
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e MPLCONFIGDIR=/tmp/matplotlib -p 8888:8888 -v "$PWD:/app" --entrypoint jupyter qdgt-nta lab --ip=0.0.0.0 --no-browser --port=8888
```

Use the URL printed by Jupyter. The notebook reads the original spreadsheet and model artifacts; it generates figures only if their PNG files are missing.

## Figures and provenance

| Output | Source | Scope |
| --- | --- | --- |
| `figure_s1_distributions.png` | Table S1 | Diffusion coefficient, molecular weight, and logP distributions |
| `figure_s7_ie_rf.png` | Table S5 | Predicted logIE versus measured logRF, with mode-specific linear fits |
| `figure_3b_ad_coverage.png` | Table S10 | AD overlap and top 20 classes; layout is an adaptation |
| `figure_4a_validation.png` | Table S11 | Measured versus predicted aqueous and DGT concentrations |
| `figure_5_predicted_concentrations.png` | Tables S14–S15 | Top 25 concentrations per ion mode, a readable subset of Figure 5 |
| `figure_s8_fold_errors.png` | Tables S11–S13 | Fold error distributions |

These figures reproduce tabulated values, not the original artwork. The model-performance panels in Figure 2, Figure 3A, Figure 4B, and the supplementary AD panels require original training/test assignments, per-sample predictions, or AD evaluations not supplied in the workbook. The structural-space panels also depend on UMAP settings and source datasets that are not fully specified here. Figure 1 and Figure S3 are conceptual diagrams. The saved models support inference, but the repository does not include the full training pipeline or original random seeds, so published model metrics cannot be regenerated exactly.

The feature code preserves the published repository's descriptor and fingerprint order. The original code standardized each molecule's RDKit descriptor vector individually; this is retained for model compatibility. The saved model pickles are trusted artifacts and should only be loaded from this repository. See `src/qdgt_nta/` for inference and plotting code.
