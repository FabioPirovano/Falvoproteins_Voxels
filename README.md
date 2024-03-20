# Prediction Flavoproteins EM using Voxels

This repository contains the script and data used for "Automatic features extraction from PDB files for flavoproteins redox potential prediction models"

## Flavoprotein Datset

> The Flavoproteins Dataset is available at `dataset.xlsx`, at sheet 'Foglio_backup'. 
For each flavoprotein used in this work we report:

```
- PDB-ID 
- Organism and Family
- Classification and Cofactor-type
- Technique resolution
- Em, E1, E2, 𝛥E
- Cofactor (if cofactor is FMN = 1, if cofactor is FAD = 0)
- Reference of the experimental work
```

<br/>

## PREPROCESS STEPS

> All PDB files contained in the Dataset have been downloaded and loaded into PyMOL:

- *1° Step*: All PDB files have been aligned based on cofactor (FMN or FAD);
- *2° Step*: Spherical cuts were made, with the center in the cofactor, of different radius (10,6..);
- *3° Step*: The cut PDB files were downloaded and saved separately, based on the cut radius;

<br/>
> In the folders



## codes

> All the scripts used to reproduce the work are here reported:

- *PREDICTION.py*: scripts for the ML pipeline used to test the performance of the ML models considered;
- *PREPROCESS.py*: features extraction of the flavoprotein PDB files reported in 'Foglio_backup' of dataset.xlsx using PyUUL (Voxels);
- *ML_models.py*: contains all the machine learning models hparams;
- *ANALYSIS.ipynb*:  scripts for the results analysis;

<br/>














