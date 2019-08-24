from __future__ import division

import numpy as np
import math
from scipy.sparse import csr_matrix

def construct_cont_table(dic):
    mat = []
    for key in dic:
        line = []
        for i in dic[key]:
            line.append(dic[key][i])
        mat.append(line)
    mat = np.transpose(np.array(mat, dtype=np.int64))
    return mat

def rand_values(cont_table):
    n = cont_table.sum()
    sum1 = (cont_table.multiply(cont_table)).sum()
    sum2 = (np.asarray(cont_table.sum(axis=1)) ** 2).sum()
    sum3 = (np.asarray(cont_table.sum(axis=0)) ** 2).sum()
    a = (sum1 - n)/2.0;
    b = (sum2 - sum1)/2
    c = (sum3 - sum1)/2
    d = (sum1 + n**2 - sum2 - sum3)/2
    return a, b, c, d

def adj_rand_index(dic):
    mat = construct_cont_table(dic)
    mat = csr_matrix(mat)
    a, b, c, d = rand_values(mat)
    nk = a+b+c+d
    return (nk*(a+d) - ((a+b)*(a+c) + (c+d)*(b+d)))/(nk**2 - ((a+b)*(a+c) + (c+d)*(b+d)))

def rand_index(dic):
    mat = construct_cont_table(dic)
    mat = csr_matrix(mat)
    a, b, c, d = rand_values(mat)
    return (a+d)/(a+b+c+d)

def calc_entropy(vector):
    h = 0.0
    # normalization
    if vector.sum() != 0:
        # normalize
        vector = vector / vector.sum()
        # remove zeros
        vector = vector[vector != 0]
        # compute h
        h = np.dot(vector, np.log2(vector) * (-1))
    return h

def entropy(dic):
    mat = construct_cont_table(dic)
    h = 0.0
    n = mat.sum()
    for i in range(0, mat.shape[0]):
        h += (mat[i,:].sum() / n) * (1 / math.log(mat.shape[1], 2) * calc_entropy(mat[i, :]))
    return h


def purity(dic):
    mat = construct_cont_table(dic)
    n = mat.sum()
    p = 0.0
    for i in range(0,mat.shape[0]):
        p += mat[i,:].max()/n
    return p
