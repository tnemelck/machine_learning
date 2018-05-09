#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 10:13:41 2018

@author: elvex
"""

import sklearn.metrics as skmet
import sklearn.ensemble as ske
import sklearn.model_selection as skms
import numpy as np

def div_dataset(Xr, Y):
    X_tr, X_te, Y_tr, Y_te = skms.train_test_split(Xr, Y)
    return (X_tr, X_te, Y_tr, Y_te)


def GB_test_dirty(Xr, Y, n = 5):
    l = []
    GB =  ske.GradientBoostingClassifier()
    for i in range(n):
        (X_tr, X_te, Y_tr, Y_te) = div_dataset(Xr, Y)
        GB.fit(X_tr, Y_tr)
        y_pred = GB.predict(X_te)
        l.append(skmet.accuracy_score(Y_te, y_pred))
    m = np.mean(np.array(l))
    print("La moyenne d'exactitude de ce classifier est de {}.".format(m))
        

