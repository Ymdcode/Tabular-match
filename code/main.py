import numpy as np
import pandas as pd
import os

pwd = os.getcwd()

df1 = pd.read_csv(pwd+'/data/train.csv')
df2 = pd.read_csv(pwd+'/data/test.csv')
sam = pd.read_csv(pwd+'/data/sample_submission.csv')

sub1 = pd.read_csv(pwd+'/other data/comparative_1.74396.csv')
sub2 = pd.read_csv(pwd+'/other data/result_2 (1.74399).csv')


def combine(main, support):
    sub1v = support.values
    sub2v = main.values
    imp = main.copy()

    impv = main.values
    NCLASS = 9

    weighs = [0.4,0.5,0.6]
    weight = np.random.choice(weighs,3)

    for i in range(len(main)):

        row1 = sub1v[i, 1:]
        row2 = sub2v[i, 1:]
        row1_sort = np.sort(row1)
        row2_sort = np.sort(row2)

        row = (row2 * weight) + (row1 * (1.0 - weight))

        for j in range(NCLASS):
            if ((row2[j] == row2_sort[8]) and (row1[j] != row1_sort[8])):
                row = row2

        impv[i, 1:] = row

    imp.iloc[:, 1:] = impv[:, 1:]
    return imp

sub_imp = combine(sub1, sub2)

sub_imp.to_csv("result.csv",index=False)