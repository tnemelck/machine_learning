#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 10:46:39 2018

@author: elvex
"""

import numpy as np
import sklearn.preprocessing as skp
import pandas as pd
#import math
import re
import sklearn.ensemble as ske
import sklearn.model_selection as skms
import matplotlib.pyplot as plt

path = "/home/elvex/.kaggle/competitions/titanic/"
path_train = path + "train.csv"
path_test = path + "test.csv"

def extract_data(file):
    bdd = pd.read_csv(file, sep=',', header=0, index_col=0)
    return bdd


def div_cabine(serie):
    l = list(serie)
    cab_lettre, cab_num, cab_nb = [], [], []
    for e in l:
        if isinstance(e, float):
            cab_lettre.append(e)
            cab_num.append(e)
            cab_nb.append(e)
        else:
            L = list(map(lambda x : ord(x), re.findall('[A-Za-z]+', e)))
            N = list(map(int, re.findall('[0-9]+', e)))
            cab_lettre.append(chr(np.mean(np.array(L)).astype(int)))
            cab_num.append(np.mean(np.array(N)))
            cab_nb.append(len(L))
    #cab_lettre = list(map(lambda x: chr(x) if x != -1 else -1, cab_lettre))
    df0 = pd.DataFrame({"cab_lettre" : cab_lettre})
    df0 = pd.get_dummies(df0, prefix= "cabine_classe")
    df1 = pd.DataFrame({
            'cab_num' : cab_num,
            'cab_nb' : cab_nb
            })
    df = pd.concat([df0, df1], axis =1)
    return df


def formate_cabine(bdd):
    cab = bdd['Cabin']
    df = div_cabine(cab)
    bdd.drop('Cabin', axis=1, inplace=True)
    bdd = pd.concat([bdd, df], axis=1, join_axes=[bdd.index])
    return bdd


def formate_ticket(bdd):
    bdd.drop('Ticket', axis = 1, inplace = True)
    return bdd


def formate_nom(bdd):
    lst = list(bdd["Name"])
    title = list(map(lambda x: re.split('[,.]', x)[1], lst))
    df = pd.DataFrame({"Title": title})
    df = pd.get_dummies(df, prefix = "Title")
    bdd.drop('Name', axis = 1, inplace = True)
    bdd = pd.concat([bdd, df], axis=1, join_axes=[bdd.index])
    return bdd

def formate_sexe(bdd):
    df = pd.get_dummies(bdd["Sex"], prefix="Sexe")
    bdd.drop("Sex", axis=1, inplace=True)
    bdd = pd.concat([bdd, df], axis=1, join_axes=[bdd.index])
    return bdd
    

def formate_embarked(bdd):
    df = pd.get_dummies(bdd["Embarked"], prefix="Embarked")
    bdd.drop("Embarked", axis=1, inplace=True)
    bdd = pd.concat([bdd, df], axis=1, join_axes=[bdd.index])
    return bdd


def formate_pclass(bdd):
    df = pd.get_dummies(bdd["Pclass"], prefix="Pclass")
    bdd.drop("Pclass", axis=1, inplace=True)
    bdd = pd.concat([bdd, df], axis=1, join_axes=[bdd.index])
    return bdd

def formate_age(bdd):
    bdd['Child'] = (bdd.Age < 8) * 1
    return bdd

def formate_SibSp(bdd):
    df = pd.get_dummies(bdd["SibSp"], prefix="SibSp")
    bdd.drop("SibSp", axis=1, inplace=True)
    bdd = pd.concat([bdd, df], axis=1, join_axes=[bdd.index])
    return bdd


def formate_parch(bdd):
    df = pd.get_dummies(bdd["Parch"], prefix="Parch")
    bdd.drop("Parch", axis=1, inplace=True)
    bdd = pd.concat([bdd, df], axis=1, join_axes=[bdd.index])
    return bdd

def formate_drop(bdd):
    bdd.drop("Fare", axis=1, inplace = True)
    bdd.drop("cab_num", axis=1, inplace = True)
    bdd.drop("cabine_class_A", axis=1, inplace = True)
    bdd.drop("cabine_class_B", axis=1, inplace = True)
    bdd.drop("cabine_class_C", axis=1, inplace = True)
    bdd.drop("cabine_class_D", axis=1, inplace = True)
    bdd.drop("cabine_class_E", axis=1, inplace = True)
    bdd.drop("cabine_class_F", axis=1, inplace = True)
    bdd.drop("cabine_class_G", axis=1, inplace = True)
    bdd.drop("cabine_class_T", axis=1, inplace = True)
    
    
    return bdd

def bdd2ary(adr):
    bdd = pd.read_csv(adr, sep=',', header=0, index_col=0)
    bdd = formate_cabine(bdd)
    bdd = formate_nom(bdd)
    bdd = formate_ticket(bdd)
    bdd = formate_embarked(bdd)
    bdd = formate_sexe(bdd)
    bdd = formate_pclass(bdd)
    bdd = formate_age(bdd)
    bdd = formate_SibSp(bdd)
    bdd = formate_parch(bdd)
    #bdd = formate_drop(bdd)
    bdd = bdd.fillna(value = np.nan)
    Y = bdd["Survived"].as_matrix()
    bddX = bdd.drop("Survived", axis=1)
    X = bddX.as_matrix()
    return (X, Y, bdd)


def formate_X(X, scaler = False):
    imp = skp.Imputer(missing_values=np.nan, strategy='median', axis=0)
    X = imp.fit_transform(X)
    if not scaler: scaler = skp.Normalizer().fit(X)
    Xr = scaler.transform(X)    
    return Xr, scaler

def div_dataset(X, Y):
    X_tr, X_te, Y_tr, Y_te = skm.train_test_split(X, Y)
    return (X_tr, X_te, Y_tr, Y_te)

def get_Data():
    X, Y, bdd = bdd2ary(path_train)
    Xr, scaler = formate_X(X)
    X_tr, X_te, Y_tr, Y_te = div_dataset(Xr, Y)
    return X_tr, X_te, Y_tr, Y_te, scaler, bdd


def test_nb_estimator(X_tr, Y_tr, min_val = 1, max_val = 200, pas = 1, nb_cv = 5, loss = "deviance"):
    """
    Max atteint en 38.0 et vaut 0.82714577377653 avec la descente de gradient par deviance
    Max atteint en 74.0 et vaut 0.8293741335781745 avec la descente de gradient par exponential
    On choisit ... exponential avec n_estimator = 74 !
    """
    x, y = [], []
    for i in range(min_val, max_val, pas):
        GB = ske.GradientBoostingClassifier(n_estimators= i, loss = loss)
        scores = skms.cross_val_score(GB, X_tr, Y_tr, cv= nb_cv)
        score = np.mean(scores)
        x.append(i)
        y.append(score)
    plt.plot(x, y, label = "Performence selon n_estimator")
    plt.legend(loc = 'best')
    plt.show()
    ary = np.array([x, y])
    print("Max atteint en {} et vaut {} avec la descente de gradient par {}".format(ary[0, np.argmax(ary[1, :])], np.max(ary[1, :]), str(loss)) )
    return ary


def test_max_depth(X_tr, Y_tr, min_val = 2, max_val = 10, pas = 1, nb_cv = 5):
    """
    Le Max atteint vaut 0.8399506228257211 avec  une profondeur d'arbre de 3.0
    On choisit donc max_depth = 3.
    """
    x, y = [], []
    for i in range(min_val, max_val, pas):
        GB = ske.GradientBoostingClassifier(loss = "exponential", n_estimators=74, max_depth=i)
        scores = skms.cross_val_score(GB, X_tr, Y_tr, cv= nb_cv)
        score = np.mean(scores)
        x.append(i)
        y.append(score)
    plt.plot(x, y, label = "Performence selon la profondeur d'arbre")
    plt.legend(loc = 'best')
    plt.show()
    ary = np.array([x, y])
    print("Le Max atteint vaut {} avec  une profondeur d'arbre de {}".format(np.max(ary[1, :]), ary[0, np.argmax(ary[1, :])]) )
    return ary 


def test_learning_rate(X_tr, Y_tr, min_val = 0.02, max_val = 0.25, pas = 0.01, nb_cv = 10):
    """
    Le Max atteint vaut 0.8259737827715355 avec  untaux d'apprentissage de 0.10999999999999999
    On choisit learning rate = 0.11
    """
    x, y = [], []
    for i in np.arange(min_val, max_val, pas):
        GB = ske.GradientBoostingClassifier(loss = "exponential", n_estimators=74, max_depth=3, learning_rate=i)
        scores = skms.cross_val_score(GB, X_tr, Y_tr, cv= nb_cv)
        score = np.mean(scores)
        x.append(i)
        y.append(score)
    plt.plot(x, y, label = "Performence selon le taux d'apprentissage")
    plt.legend(loc = 'best')
    plt.show()
    ary = np.array([x, y])
    print("Le Max atteint vaut {} avec  untaux d'apprentissage de {}".format(np.max(ary[1, :]), ary[0, np.argmax(ary[1, :])]) )
    return ary 

def test_min_sample(X_tr, Y_tr, min_val = 2, max_val = 200, pas = 1, nb_cv = 5):
    """
    Le Max atteint vaut 0.8260411513351895 avec  un nb min avant séparation de 65.0
    On choisit min_sample_split = 65
    """
    x, y = [], []
    for i in np.arange(min_val, max_val, pas):
        GB = ske.GradientBoostingClassifier(loss = "exponential", n_estimators=74, max_depth=3,
                                            learning_rate=0.11, min_samples_split= i)
        scores = skms.cross_val_score(GB, X_tr, Y_tr, cv= nb_cv)
        score = np.mean(scores)
        x.append(i)
        y.append(score)
    plt.plot(x, y, label = "Performence selon le nb min avant séparation d'une branche.")
    plt.legend(loc = 'best')
    plt.show()
    ary = np.array([x, y])
    print("Le Max atteint vaut {} avec  un nb min avant séparation de {}".format(np.max(ary[1, :]), ary[0, np.argmax(ary[1, :])]) )
    return ary 


def test_max_features(X_tr, Y_tr, min_val = 1, max_val = False, pas = 1, nb_cv = 5):
    """
    Le Max atteint vaut 0.8260349451926687 avec  un nb min avant séparation de 31.0
    On choisit max_features = 31
    """
    max_val = X_tr.shape[1] if not max_val else min(max_val, X_tr.shape[1])
    x, y = [], []
    GB = ske.GradientBoostingClassifier(loss = "exponential", n_estimators=74, max_depth=3,
                                        learning_rate=0.11, min_samples_split=65,
                                        max_features= None)
    scores = skms.cross_val_score(GB, X_tr, Y_tr, cv= nb_cv)
    score = np.mean(scores)
    x.append(-1)
    y.append(score)
    for i in np.arange(min_val, max_val, pas):
        GB = ske.GradientBoostingClassifier(loss = "exponential", n_estimators=74, max_depth=3,
                                            learning_rate=0.11, min_samples_split=65,
                                            max_features= i)
        scores = skms.cross_val_score(GB, X_tr, Y_tr, cv= nb_cv)
        score = np.mean(scores)
        x.append(i)
        y.append(score)
    plt.plot(x, y, label = "Performence selon le nb max de features.")
    plt.legend(loc = 'best')
    plt.show()
    ary = np.array([x, y])
    ind = ary[0, np.argmax(ary[1, :])]
    val = ary[1, int(ind)]
    ind = None if ind == -1 else ind
    print("Le Max atteint vaut {} avec  un nb max de features de {}".format(val, ind) )
    return ary 

def clf_importance(bdd, clf):
    imp = clf.feature_importances_
    ind = np.argsort(imp)[::-1]
    plt.title("Importances des caractéristiques")
    for tree in clf.estimators_:
        #plt.plot(range(bdd.shape[1]), tree[ind], 'r')
        plt.plot(range(bdd.shape[1]), imp[ind], 'b')
    plt.show()
    for f in range(bdd.shape[1]):
        print("{} feature : {} {}".format(f+1, bdd.columns[ind[f]], imp[ind[f]]))
        
def plot_importance(bdd, feature, bins=20):
    survived = bdd[bdd.Survived == 1]
    dead = bdd[bdd.Survived == 0]
    x1 = np.array(dead[feature].dropna())
    x2 = np.array(survived[feature].dropna())
    plt.hist([x1, x2], label = ["survived", "dead"], bins = bins, color = ['b', 'k'], normed=[True, True])
    plt.legend(loc="upper_left")
    plt.title("Distribution de la feature {}".format(feature))
    plt.show()
    
    



#def manip_data(bdd):
    