#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 17:38:59 2024

@author: fabiopirovano
"""

import pandas as pd
import numpy as np
import statistics

n_repetition = 3
train_test_split = 5
res = 1
radius = 16

models = ["WeightedEnsemble", "XGBoost", "CatBoost", "RandomForest"]

models_performance = pd.DataFrame(index=models, columns=["MAE", "RMSE", "Spearman Correlation"], data=0)
models_performance_without_validation = pd.DataFrame(index=models, columns=["MAE", "RMSE", "Spearman Correlation"], data=0)

# Inizializza un dizionario per raccogliere i risultati per ogni modello
risultati_modello = {"MAE": {}, "RMSE": {}, "Spearman Correlation": {}}
risultati_modello_without_validation = {"MAE": {}, "RMSE": {}, "Spearman Correlation": {}}

for i in range(3):
    df_orig = pd.read_excel("TUNING_sfera_" + str(radius) + "_res_" + str(res) + "_new.xlsx", index_col=0,
                            sheet_name="Iterazione_" + str(i + 1))

    df_orig = df_orig.rename(columns={'Unnamed: 2': 'train test split'})
    df_orig["model"] = df_orig["Modello"].apply(lambda x: x.split("_")[0])

    df = df_orig[df_orig["Modello"].str.contains("FULL") == False]
    df_FULL = df_orig[df_orig["Modello"].str.contains("FULL") == True]

    models = list(set(df["model"].values))
    models_winner = dict()

    for model in models:
        df2 = df[df["model"] == model]
        df3 = df2.groupby(by="train test split")["Score Validation"].agg(lambda x: max(x)).values

        for i in range(5):
            df4 = df2[df2["Score Validation"] == df3[i]]["Modello"]
            models_winner[model + "_" + str(i)] = df4.values[0]

    values = dict()
    values_spearman = dict()
    values_rmse = dict()

    for model in models:
        df2 = df[df["model"] == model]
        values[model] = 0
        values_rmse[model] = 0
        values_spearman[model] = 0

        for i in range(5):
            df3 = df2[df2["Modello"] == models_winner[model + "_" + str(i)]]
            values[model] += df3["MAE"].mean()
            values_rmse[model] += df3["RMSE"].mean()
            values_spearman[model] += df3["Spearman Correlation"].mean()
        values[model] = values[model] / train_test_split
        values_rmse[model] = values_rmse[model] / train_test_split
        values_spearman[model] = values_spearman[model] / train_test_split

    for model in models:
        models_performance_without_validation.loc[model, "MAE"] += values[model] / n_repetition
        models_performance_without_validation.loc[model, "RMSE"] += values_rmse[model] / n_repetition
        models_performance_without_validation.loc[model, "Spearman Correlation"] += values_spearman[model] / n_repetition

        # Memorizza i risultati per ogni modello e metrica senza validazione
        risultati_modello_without_validation["MAE"].setdefault(model, []).append(values[model])
        risultati_modello_without_validation["RMSE"].setdefault(model, []).append(values_rmse[model])
        risultati_modello_without_validation["Spearman Correlation"].setdefault(model, []).append(values_spearman[model])

    for model_winner in models_winner.keys():
        df_winner = df_FULL[df_FULL["Modello"] == models_winner[model_winner] + "_FULL"]
        df_winner = df_winner[df_winner["train test split"].astype(str) == model_winner.split("_")[1]]

        models_performance.loc[model_winner.split("_")[0], "MAE"] += df_winner["MAE"].values[0]
        models_performance.loc[model_winner.split("_")[0], "RMSE"] += df_winner["RMSE"].values[0]
        models_performance.loc[model_winner.split("_")[0], "Spearman Correlation"] += df_winner[
            "Spearman Correlation"].values[0]
       
        # Memorizza i risultati per ogni modello e metrica con validazione
        risultati_modello["MAE"].setdefault(model_winner, []).append(df_winner["MAE"].values[0])
        risultati_modello["RMSE"].setdefault(model_winner, []).append(df_winner["RMSE"].values[0])
        risultati_modello["Spearman Correlation"].setdefault(model_winner, []).append(df_winner["Spearman Correlation"].values[0])

for model in models:
    models_performance.loc[model, :] /= (train_test_split * n_repetition)


# Calcola la deviazione standard e stampa i risultati per ciascun modello e metrica senza validazione
for metrica in ["MAE", "RMSE", "Spearman Correlation"]:
    print(f"\nDeviazione standard per {metrica} senza validazione:")
    for model in models:
        if model in risultati_modello_without_validation[metrica]:
            deviazione_standard = statistics.stdev(risultati_modello_without_validation[metrica][model])
            print(f"{model}: {deviazione_standard}")
        else:
            print(f"{model}: Nessun risultato registrato per {metrica}")

# Calcola la deviazione standard e stampa i risultati per ciascun modello e metrica con validazione
for metrica in ["MAE", "RMSE", "Spearman Correlation"]:
    print(f"\nDeviazione standard per {metrica} con validazione:")
    for model in models:
        if model_winner in risultati_modello[metrica]:
            deviazione_standard = statistics.stdev(risultati_modello[metrica][model_winner])
            print(f"{model}: {deviazione_standard}")
        else:
            print(f"{model}: Nessun risultato registrato per {metrica}")