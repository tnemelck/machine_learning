#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 15:25:51 2018

@author: elvex
"""

import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as skp
import sklearn.model_selection as skms


def get_BDD(contenu = "train"):
    path = "/home/elvex/.kaggle/competitions/titanic/"
    path_train = path + "train.csv"
    path_test = path + "test.csv"
    dico_path = {"train" : path_train, "test": path_test}
    adr = dico_path.get(contenu, path_test)
    bdd = pd.read_csv(adr, sep=',', header=0, index_col=0)
    return bdd


def plot_importance(bdd, feature, bins=20, range=None):
    survived = bdd[bdd.Survived == 1]
    dead = bdd[bdd.Survived == 0]
    x2 = np.array(dead[feature].dropna())
    x1 = np.array(survived[feature].dropna())
    plt.hist([x1, x2], label = ["survived", "dead"], bins = bins, color = ['b', 'k'], normed=[True, True])
    plt.legend(loc="upper_left")
    plt.title("Distribution de la feature {}".format(feature))
    plt.show()
    
    
def plot_importance_continue(bdd, feature):
    survived = bdd[bdd.Survived == 1]
    dead = bdd[bdd.Survived == 0]
    x1 = np.array(dead[feature].dropna())
    x2 = np.array(survived[feature].dropna())
    h_D = list(np.histogram(x1, bins=100, range = (0, 100)))
    h_D[0] = h_D[0] / x1.size
    h_V = list(np.histogram(x2, bins = 100, range = (0, 100)))
    h_V[0] = h_V[0] / x2.size
    plt.figure()
    plt.subplot(211)
    plt.plot(h_V[1][:-1], h_V[0], color = 'b', label = "Vivant")
    plt.plot(h_D[1][:-1], h_D[0], color = 'r', label = "Mort")
    plt.title("Distribution de la feature {}".format(feature))
    plt.legend(loc="upper_left")
    plt.subplot(212)
    DH = h_V[0] - h_D[0]
    plt.plot(h_V[1][:-1], DH, color = 'k', label = "Vivant-Mort")
    plt.legend(loc="upper_left")
    plt.show()
    return h_V, h_D, DH
    
    


def plot_importance_all(bdd, bins=20):
    for c in bdd.columns: plot_importance(bdd, c, bins)
    
    
def correlation(bdd, seuil = 0.05):
    mtx_cor = np.corrcoef(bdd.as_matrix(), rowvar=0)[0]
    for i in range(1, mtx_cor.size):
        deb = "!!! " if abs(mtx_cor[i]) < seuil else ""
        print("{}La corrélation entre la survie et la variable '{}' vaut {}".format(deb, bdd.columns[i], mtx_cor[i]))
    return mtx_cor
    

def formate_Pclass(bdd):
    df = bdd["Pclass"]
    df.fillna(df.median(), inplace = True)
    df = pd.get_dummies(df, prefix = "Pclass")
    bdd.drop('Pclass', axis = 1, inplace = True)
    bdd = pd.concat([bdd, df], axis=1, join_axes=[bdd.index])
    return bdd


def formate_Name(bdd):
    df = bdd["Name"]
    df = df.map(lambda x: re.split('[,.]', x)[1])
    df = pd.get_dummies(df, prefix = "Title")
    bdd.drop('Name', axis = 1, inplace = True)
    bdd = pd.concat([bdd, df], axis=1, join_axes=[bdd.index])
    return bdd
    


def formate_Sex(bdd):
    df = bdd["Sex"]
    #df.fillna(df.median(), inplace = True)
    df = pd.get_dummies(df, prefix = "Sexe")
    bdd.drop('Sex', axis = 1, inplace = True)
    bdd = pd.concat([bdd, df], axis=1, join_axes=[bdd.index])
    return bdd


def formate_Age(bdd):
    df = bdd["Age"]
    df.fillna(df.median(), inplace = True)
    df_young_child = pd.DataFrame({"Young child" : (df <= 5) * 1})
    df_old_teen = pd.DataFrame({"Old Teen" : ((df >= 18) * (df <= 21)) * 1})
    df_young_adult = pd.DataFrame({"Young adult" : ((df > 21) * (df <= 30)) * 1})
    df_elder = pd.DataFrame({"Elder" : (df >= 64) *1})
    bdd.drop('Age', axis = 1, inplace = True)
    bdd = pd.concat([bdd, df, df_young_child, df_old_teen, df_young_adult, df_elder], axis=1, join_axes=[bdd.index])
    return bdd


def formate_SibSp_Parch(bdd):
    df1 = bdd["SibSp"]
    df2 = bdd["Parch"]
    df1.fillna(df1.median())
    df2.fillna(df2.median())
    dfS = pd.DataFrame({"Family": df1+df2})
    dfA = pd.DataFrame({"Alone" : (dfS.Family == 0) * 1})
    bdd.drop('SibSp', axis = 1, inplace = True)
    bdd.drop('Parch', axis = 1, inplace = True)
    bdd = pd.concat([bdd, df1, df2, dfS, dfA], axis=1, join_axes=[bdd.index])
    return bdd


def map_ticket(s):
    s = " ".join(s.split(" ")[:-1])
    s = s.replace(".", "")
    s = s.replace("/", "")
    s = s.replace(" ", "")
    return s

def formate_Ticket(bdd):
    bdd.drop('Ticket', axis = 1, inplace = True)
    return bdd


def formate_Fare(bdd):
    df = bdd["Fare"]
    df.fillna(df.median(), inplace = True)
    bdd.drop('Fare', axis = 1, inplace = True)
    bdd = pd.concat([bdd, df], axis=1, join_axes=[bdd.index])
    return bdd


def formate_Cabin(bdd):
    bdd.drop('Cabin', axis = 1, inplace = True)
    return bdd


def formate_Embarked(bdd):
    df = bdd["Embarked"]
    #df.fillna(df.median(), inplace = True)
    df = pd.get_dummies(df, prefix = "Embarked")
    bdd.drop('Embarked', axis = 1, inplace = True)
    bdd = pd.concat([bdd, df], axis=1, join_axes=[bdd.index])
    return bdd

def drop(bdd, feature):
    try:
        bdd = bdd.drop(feature, axis = 1, inplace=False)
    except ValueError:
        pass
    return bdd


def formate_drop(bdd):
    liste_drop = ['Embarked_Q', 'Family', 'SibSp', 'Title_ the Countess', 'Title_ the Countess',
                  'Title_ Sir', 'Title_ Ms', 'Title_ Major', 'Title_ Lady', 'Title_ Jonkheer',
                  'Title_ Dr', 'Title_ Don', 'Title_ Col', 'Title_ Capt']
    for d in liste_drop:
        bdd = drop(bdd, d)
    return bdd
    


def formate_bdd(BDD):
    bdd = BDD.copy(deep = True)
    bdd = formate_Pclass(bdd)
    bdd = formate_Name(bdd)
    bdd = formate_Sex(bdd)
    bdd = formate_Age(bdd)
    bdd = formate_SibSp_Parch(bdd)
    bdd = formate_Ticket(bdd)
    bdd = formate_Fare(bdd)
    bdd = formate_Cabin(bdd)
    bdd = formate_Embarked(bdd)
    bdd = formate_drop(bdd)
    bdd = bdd.fillna(bdd.median())
    return bdd

def bdd2ary(bdd, scaler = None):
    bdd = bdd.fillna(value = np.nan)
    Y = bdd["Survived"].as_matrix()
    bddX = bdd.drop("Survived", axis=1)
    X = bddX.as_matrix()
    imp = skp.Imputer(missing_values=np.nan, strategy='median', axis=0)
    X = imp.fit_transform(X)
    if not scaler: scaler = skp.Normalizer().fit(X)
    Xr = scaler.transform(X)    
    return (Xr, Y, scaler)