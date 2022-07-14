import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import optuna
import warnings
from tqdm import tqdm


# For Machine learning =========================================
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score


# XGBoost ==============================
from xgboost import XGBClassifier
np.set_printoptions(precision=3)
warnings.simplefilter('ignore')


# Set Path2Corpus =========================
path_to_base = `PATH_TO_BASE_DIRECTORY'
version = 'INPUT_YOUR_VERSION'
os.chdir(path_to_base)


# Set variables ==============================
class_names = np.array(['U','NU'])
cvs = 10 # cv num
n_trials = 10


# Data IO ------------------------------------------
def pkl_loader(pkl_filename):
    with open(pkl_filename, 'rb') as web:
        data = pickle.load(web)
    return data


def pkl_saver(pkl_filename, object):
    with open(pkl_filename, 'wb') as web:
        pickle.dump(object , web)


def save_result(cv_results, class_names, eval):
    results_list = pd.DataFrame(cv_results)
    results_list.columns = ['id', 'obs', 'pred', *class_names.tolist()]
    results_list.to_csv('xgb_results_list' + eval + '.csv')
    cfm = pd.DataFrame(confusion_matrix(results_list.iloc[:,1], results_list.iloc[:,2], labels=class_names), columns=class_names, index=class_names)
    report = classification_report(results_list.iloc[:,1], results_list.iloc[:,2], target_names=class_names)
    with open('summary' + eval + '.txt','w') as res:
        res.write(report)
    cfm.to_csv('confusion_matrix' + eval + '.csv')


# Machine Learning =========================
def train_test_splitter(df, cvs, cv):
    idx = list(df.index)
    test_idx = list(idx[cv::10])
    train_idx = set(idx)-set(test_idx)
    df_test = df.loc[test_idx]
    df_train = df.loc[train_idx]
    X_train = df_train.drop(df.columns[0], axis=1).values
    X_test = df_test.drop(df.columns[0], axis=1).values
    y_train = df_train[df.columns[0]]
    y_test = df_test[df.columns[0]]
    return df_train, df_test, X_train, X_test, y_train, y_test


def opt_xgb(trial):
    n_estimators = trial.suggest_int('n_estimators', 0, 1000)
    max_depth = trial.suggest_int('max_depth', 1, 20)
    min_child_weight = trial.suggest_int('min_child_weight', 1, 20)
    subsample = trial.suggest_discrete_uniform('subsample', 0.5, 0.9, 0.1)
    colsample_bytree = trial.suggest_discrete_uniform('colsample_bytree', 0.5, 0.9, 0.1)
    xgboost_tuna = XGBClassifier(
        objective = 'binary:logistic',
        random_state=42,
        nthread = 8,
        n_estimators = n_estimators,
        max_depth = max_depth,
        min_child_weight = min_child_weight,
        subsample = subsample,
        colsample_bytree = colsample_bytree,
    )
    xgboost_tuna.fit(X_val_train, y_val_train)
    tuna_pred_val = xgboost_tuna.predict(X_val)
    return (1.0 - (accuracy_score(y_val, tuna_pred_val)))


def top_n_accuracy(preds, truths, n, class_names):
    best_n = class_names[np.argsort(preds, axis=1)[:,-n:]]
    ts = np.array(truths).reshape(-1,1)
    successes = 0
    for i in range(ts.shape[0]):
      if ts[i] in best_n[i,:]:
        successes += 1
    return float(successes)/ts.shape[0]


# XGBoost =========================================
# Model and DataFrame construction ========================
df_plain = pd.read_csv('PATH_TO_DATAFRAME_2005')
'''
This dataframe has 5 columns which are 'urbin','b2','b3','b4', and 'b5' sequentially. 
First column, 'urbin', contained the binary land cover type which are U and NU from the ground truth data. 
The next 4 columns contained the DNs of each pixel in band 2, band 3, band 4, and band 5 extracted from LANDSAT 5TM 2005. 
'''


# Cross Validation Loop ========================
param_names = np.array(['n_estimators','max_depth','min_child_weight','subsample','colsample_bytree']).reshape(1,-1)
val_params_memory = np.zeros((cvs, param_names.shape[1]))
test_params_memory = np.zeros((cvs, param_names.shape[1]))
for outer_cv in tqdm(range(0, cvs)):
    print('outer_cv' + str(outer_cv) + '_processing...')
    df_train, df_test, X_train, X_test, y_train, y_test = train_test_splitter(df, cvs, outer_cv)
    # Grid search tuning ====================
    for inner_cv in range(0, cvs):
        print('inner_cv' + str(inner_cv) + '_processing...')
        _, _, X_val_train, X_val, y_val_train, y_val = train_test_splitter(df_train, cvs, inner_cv)
        study = optuna.create_study()
        study.optimize(opt_xgb, n_trials=n_trials)
        for param in range(len(param_names[0])):
            val_params_memory[inner_cv, param] = study.best_params[param_names[0,param]]
    val_mean_best_params = {name: param for name, param in zip(param_names[0], np.mean(val_params_memory, axis=0))}
    for param in range(len(param_names[0])):
        test_params_memory[outer_cv, param] = val_mean_best_params[param_names[0,param]]
    # best_params setting ==================
    best_xgboost = XGBClassifier(
        objective = 'binary:logistic',
        nthread = 8, 
        random_state=42,
        n_estimators= int(val_mean_best_params['n_estimators']),
        max_depth=int(val_mean_best_params['max_depth']),
        min_child_weight=int(val_mean_best_params['min_child_weight']),
        subsample=val_mean_best_params['subsample'],
        colsample_bytree=val_mean_best_params['colsample_bytree'],
    )
    # Best_model_training ===================
    best_xgboost.fit(X_train, y_train, verbose=True)
    # Test data prediction
    y_train_pred = best_xgboost.predict(X_train)
    y_pred = best_xgboost.predict(X_test)
    y_pred_softmax = best_xgboost.predict_proba(X_test)
    # Result memorizing ========================
    try:
        train_accuracies = np.append(train_accuracies, accuracy_score(y_train, y_train_pred))
        val_top1_accuracies = np.append(val_top1_accuracies, accuracy_score(y_test, y_pred))
        val_top3_accuracies = np.append(val_top3_accuracies, top_n_accuracy(y_pred_softmax, y_test, 3, class_names))
        train_f1_scores = np.append(train_f1_scores, f1_score(y_train, y_train_pred, average='weighted'))
        c = np.append(val_f1_scores, f1_score(y_test, y_pred, average='weighted'))
    except:
        train_accuracies = accuracy_score(y_train, y_train_pred)
        val_top1_accuracies = accuracy_score(y_test, y_pred)
        val_top3_accuracies = top_n_accuracy(y_pred_softmax, y_test, 3, class_names)
        train_f1_scores = f1_score(y_train, y_train_pred, average='weighted')
        val_f1_scores = f1_score(y_test, y_pred, average='weighted')
    cv_result = np.concatenate((np.array(df_test.index).reshape(-1,1), np.array(y_test).reshape(-1,1), y_pred.reshape(-1,1), y_pred_softmax), axis=1)
    print('the num of data' + str(len(cv_result)))
    try:
        cv_results = np.concatenate((cv_results, cv_result), axis=0)
    except:
        cv_results = cv_result
    print('mean_top1_accuracy:' + str(np.mean(val_top1_accuracies)))
    print('mean_top3_accuracy:' + str(np.mean(val_top3_accuracies)))
    eval = str(np.mean(val_top1_accuracies))[2:4] + str(np.mean(val_top3_accuracies))[2:4]
    
    
# Saving results =======================================
save_result(cv_results, class_names, eval)


#Join the result with dropped columns for full bin data 2005
df_classifull2005bin = pd.DataFrame(pd.np.column_stack([cv_results, df_plain]))
df_classifull2005bin.columns = ['id', 'obs', 'pred', 'U_prob', 'NU_prob',  'keyid', 'cellid', 'gt', 'urbin','bqa','b2', 'b3', 'b4', 'b5']
df_classifull2005bin.to_csv('df_classifull2005bin.csv')
'''
We concatenated the main dataframe and predicted result to check the accuracy performance for each pixel as shown in code line 197
'''


# XGB Best_model_training ========================
mean_best_params = {name: param for name, param in zip(param_names[0], np.mean(test_params_memory, axis=0))}
best_xgboost = XGBClassifier(
    objective = 'binary:logistic',
    random_state=42,
    n_estimators= int(mean_best_params['n_estimators']),
    max_depth=int(mean_best_params['max_depth']),
    min_child_weight=int(mean_best_params['min_child_weight']),
    subsample=mean_best_params['subsample'],
    colsample_bytree=mean_best_params['colsample_bytree'],
)
X_best_train = df.drop(['urbin'], axis=1)
y_best_train = df['urbin']
best_xgboost.fit(X_best_train, y_best_train)


pkl_saver('xgb_bestmodel.binaryfile', best_xgboost)
best_xgboost = pkl_loader('xgb_bestmodel.binaryfile')


#for predict unknown dataset of 2005 ________________
X_test2005 = pd.read_csv('PATH_TO_UNKNOWN_DATAFRAME2005')
y_pred2005 = best_xgboost.predict(X_test2005)
y_pred2005smx = best_xgboost.predict_proba(X_test2005)


#for 2005 prediction with best model
df_pred2005ori = pd.DataFrame(pd.np.column_stack([y_pred2005, y_pred2005smx, X_test2005]))
df_pred2005ori.to_csv('df_pred2005ori.csv')


#for predict unknown dataset of 1999 ________________
X_test1999 = pd.read_csv('PATH_TO_UNKNOWN_DATAFRAME1999')
y_pred1999 = best_xgboost.predict(X_test1999)
y_pred1999smx = best_xgboost.predict_proba(X_test1999)


#for 1999 prediction with best model
df_pred1999ori = pd.DataFrame(pd.np.column_stack([y_pred1999, y_pred1999smx, X_test1999]))
df_pred1999ori.to_csv('df_pred1999ori.csv')


#for predict unknown dataset of 2011 ________________
X_test2011 = pd.read_csv('PATH_TO_UNKNOWN_DATAFRAME2011')
y_pred2011 = best_xgboost.predict(X_test2011)
y_pred2011smx = best_xgboost.predict_proba(X_test2011)


#for 2011 prediction with best model
df_pred2011ori = pd.DataFrame(pd.np.column_stack([y_pred2011, y_pred2011smx, X_test2011]))
df_pred2011ori.to_csv('df_pred2011ori.csv')
