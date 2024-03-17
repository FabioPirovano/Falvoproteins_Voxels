#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 17:30:53 2024

@author: fabiopirovano
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
from autogluon.tabular import TabularDataset, TabularPredictor

file_path = "df_spherex_16_res_1.csv"
excel_path = "TUNING_sfera_16_res_1_TUNING_ITERAZIONI.xlsx"
df = TabularDataset(file_path)

label = "Em" #colonna_target
n_splits = 5 # Fold nella cross-validation
n_repeat = 3 # KFold
results_list = []

for j in range(n_repeat):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=j)  

    # DataFrame per memorizzare i risultati
    results = {
        'Iterazione': [],
        'Modello': [],
        'Split': [],
        'Score Test': [],
        'Score Validation': [],
        'MAE': [],
        'RMSE': [],
        'Spearman Correlation': []}
   
    hyperparam_RF = [{'max_depth': m, 'n_estimators': n, 'max_features': o} 
                     for m in [3, 4, 5] for n in [100, 150, 200]
                     for o in ["auto", "sqrt", "log2"]]
    hyperparam_XGB = [{'max_depth': m, 'n_estimators': n, 'min_child_weight': o,
                       'learning_rate': p} 
                      for m in [3, 4, 5]
                      for n in [100, 150, 200] for o in [1, 5, 10] 
                      for p in [0.01, 0.1, 0.2, 0.4]]
    hyperparam_CAT = [{'learning_rate': m, 'depth': n}
                      for m in [0.1, 0.2, 0.4] for n in [3, 4, 5]]

    for train_index, val_index in kf.split(df):
        train_data, val_data = df.iloc[train_index], df.iloc[val_index]

        predictor = TabularPredictor(label=label,
                                     problem_type='regression',
                                     eval_metric='mean_absolute_error',
                                     verbosity=0).fit(train_data, 
                                                      hyperparameters={'CAT': hyperparam_CAT,
                                                                       'XGB': hyperparam_XGB,
                                                                       'RF': hyperparam_RF})

        y_true = val_data[label]

        for model_name, model_info in predictor.leaderboard(val_data, 
                                                            silent=True).iterrows():
            model_name = model_info['model']
            score_test = model_info['score_test']
            score_val = model_info['score_val']

            y_pred_model = predictor.predict(val_data.drop(columns=[label]),model=model_name)
            mae_model = mean_absolute_error(y_true, y_pred_model)
            rmse_model = np.sqrt(mean_squared_error(y_true, y_pred_model))
            spearman_corr_model, _ = spearmanr(y_true, y_pred_model)

            results['Iterazione'].append(j + 1)
            results['Modello'].append(model_name)
            results['Split'].append(train_index.tolist())  
            results['Score Test'].append(score_test)
            results['Score Validation'].append(score_val)
            results['MAE'].append(mae_model)
            results['RMSE'].append(rmse_model)
            results['Spearman Correlation'].append(spearman_corr_model)

    # DataFrame per ogni iterazione
    iter_results_df = pd.DataFrame(results)

    # Check file
    if not os.path.isfile(excel_path):
        with pd.ExcelWriter(excel_path, mode='w', engine='openpyxl') as writer:
            iter_results_df.to_excel(writer, sheet_name=f'Iterazione_{j+1}', index=False)
    else:
        with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl') as writer:
            iter_results_df.to_excel(writer, sheet_name=f'Iterazione_{j+1}', index=False)

print(pd.read_excel(excel_path, sheet_name=None))