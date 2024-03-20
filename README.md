# Prediction Flavoproteins Em using Voxels

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

## Voxels

In the work in question, we used Voxels to go and extract flavoprotein-related features.


> Voxels (volumetric picture elements) are the three-dimensional equivalent of pixels.
  PyUUL was used to generate the Voxels from the PDB files.
> PyUUL is a Python library designed to process 3D-structures of macromolecules, such as PDBs, 
translating them into fully differentiable data structures. 


<br/>

## PREPROCESS STEPS

> All PDB files contained in the Dataset have been downloaded and loaded into PyMOL:


- *1° Step*: all PDB files have been aligned based on cofactor (FMN or FAD);
- *2° Step*: spherical cuts were made, with the center in the cofactor, of different radius (10, 12, 14 and 16 Å);
- *3° Step*: the cut PDB files were downloaded and saved separately, based on the cut radius;

<br/>


## Folders

> Below is the explanation of the folders:


- *CUT_N_SFERA*: in the 'CUT_N_SFERA' folders you will find the processed PDB files, with 'N' having the value 10, 12, 14 and 16 Å;
- *RISULTATI/INPUT*: in this folder are located the .csv files relating to the eatures extraction for each combination of radius,
   used during cutting, and resolution's value of PyUUL;
- *RISULTATI/OUTPUT*: in this folder are located the .xlsx files used to predict the 'Em' value, with the evaluation metrics used in the Pipeline;

<br/>


## Codes

> All the scripts used to reproduce the work are here reported:

- *PREDICTION.py*: scripts for the ML pipeline used to test the performance of the ML models considered;
- *PREPROCESS.py*: scripts for features extraction of the flavoprotein PDB files reported in 'Foglio_backup' of dataset.xlsx using PyUUL (Voxels);
- *ANALYSIS.ipynb*: scripts for the results analysis;
- *ML_models.py*: contains all the machine learning models hparams;

<br/>














