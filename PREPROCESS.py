#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 17:06:10 2024

@author: fabiopirovano
"""

from pyuul import VolumeMaker # the main PyUUL module
from pyuul import utils # the PyUUL utility module
import matplotlib.pyplot as plt
import openpyxl
import urllib.request
import pandas as pd
import time,os,urllib # some standard python modules we are going to use
import matplotlib
import torch
from tqdm import tqdm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Chain import Chain
import warnings
import itertools
warnings.filterwarnings('ignore')


path="CUT_16_SFERA" #modificare in base al df usato

path_dir=""
df = pd.read_excel(path_dir+'dataset.xlsx', sheet_name = 'Foglio_backup')  
Colonna_PDB = df.iloc[:, 1] 

#PREPARO INDICE - colonna EM - colonna pH
files = sorted([f for f in os.listdir(path) if f.endswith(".pdb")])
df_PDB = pd.DataFrame({
    "PDB": [f[:-4] for f in files]})
nome_file = "dataset.xlsx"
nome_foglio = "Foglio_backup"
indici_colonne = [1, 8, 9]
df_pH_Em = pd.read_excel(nome_file, sheet_name=nome_foglio, usecols=indici_colonne)
mean_pH = df_pH_Em["pH"].mean() #media valori pH
df_pH_Em["pH"].fillna(mean_pH, inplace=True)

#START PYUUL MODULE
coords, atname = utils.parsePDB("FILE_CUT_5/", bb_only = True) #coordinates and atom names
atoms_channel = utils.atomlistToChannels(atname) #channel of each atom
radius = utils.atomlistToRadius(atname) #radius of each atom
device = "cpu"
VoxelsObject = VolumeMaker.Voxels(device=device,sparse=True)
coords = coords.to(device)
radius = radius.to(device)
atoms_channel = atoms_channel.to(device)

#VOXELS

VoxelRepresentation = VoxelsObject(coords, radius, atoms_channel, 
                                   resolution=1, 
                                   cubes_around_atoms_dim=5)

my_tensor = VoxelRepresentation.to_dense()
reshaped_tensor = my_tensor.view(my_tensor.size(0), -1)
df = pd.DataFrame(reshaped_tensor.numpy())
new_columns = [f'Features_{i}' for i in range(df.shape[1])]
df.columns = new_columns

colonne_eliminare= df.columns[df.sum()==0] #features non impattanti
df_vox_temp = df.drop(columns=colonne_eliminare)
df_vox = pd.concat([df_vox_temp, df_PDB], axis=1)

#il df_vox non contiene i PDB analizzati a pH diversi, allora li aggiungo
nuovi_nomi = ["1B1C", "1B1C", "1B1C", "1B4V", "1C0L", "1C0L", "1C0L", 
              "1FNB", "1FX1", "1GJR", "1GJR", "1J8Q", "1KIF", "1SIQ", 
              "1SIQ", "1YOB", "3GYI", "3GYJ", "3QFS", "3QFS", "3QFS", 
              "5K9B", "5K9B"]

# Seleziono solo le righe con i nomi corrispondenti
nuovi_dati_vox = df_vox[df_vox["PDB"].isin(nuovi_nomi)]
df_nuovi_vox = pd.DataFrame(columns=df_vox.columns)
for nome in nuovi_nomi:
    riga = nuovi_dati_vox[nuovi_dati_vox["PDB"] == nome].iloc[0]
    df_nuovi_vox.loc[len(df_nuovi_vox)] = riga

#Ottengo df completo
df_vox_1 = pd.concat([df_vox, df_nuovi_vox], axis=0, ignore_index=True)
df_vox_2 = df_vox_1.sort_values(by='PDB').reset_index(drop=True)
df_vox_2['Em'] = df_pH_Em['Em']
df_vox_2['pH'] = df_pH_Em['pH']
df_voxel = df_vox_2.set_index('PDB')

# Save the DataFrame as a CSV 
csv_file_path = os.path.join( "df_spherex_16_res_1_COFACTOR.csv")
df_voxel.to_csv(csv_file_path, index=True)