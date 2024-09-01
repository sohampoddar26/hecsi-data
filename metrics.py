import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import sys
import json
import os
import csv

def custom_metrics(ground, preds):
    ## inputs should be list of np arrays. 
    ## Each array is of size #tokens and contains 0s and 1s

    f1 = [f1_score(l, p, average='macro', zero_division=0) for l, p in list(zip(ground, preds))]
    pr = [precision_score(l, p, average='macro', zero_division=0) for l, p in list(zip(ground, preds))]
    re = [recall_score(l, p, average='macro', zero_division=0) for l, p in list(zip(ground, preds))]

    jacc = [jaccard_score(l, p, zero_division=0) for l, p in list(zip(ground, preds))]
    
    tokf1 = np.nanmean(f1) * 100
    tokrec = np.nanmean(re) * 100
    tokpre = np.nanmean(pr) * 100
    jaccscore = np.nanmean(jacc) * 100

    # return f1, jacc

    return tokpre, tokrec, tokf1, jaccscore



def read_input_file(filepath):
    with open(filepath) as fp:
        data = json.load(fp)

    vectors = []
    for row in data:
        tmp = np.zeros(len(row['text_tokens']), dtype=int)
        for cl in row['claims']:
            tmp[cl['start'] : cl['end']] = 1

        vectors.append(tmp)
    return vectors


def read_preds_file(filepath, ground):
    with open(filepath) as fp:
        data = json.load(fp)

    assert len(data) == len(ground), "Number of rows do not match!"

    vectors = []
    for i, (row, orig) in enumerate(zip(data, ground)):
        row = np.array(row, dtype=int)

        if not len(row) >= len(orig): print("Predictions size mismatch in row %d: %d and %d"%(i, len(row), len(orig)) )
        assert ((row==0) | (row==1)).all(), "Predictions contain items other than 0/1 in row %d"%(i)

        vectors.append(row[:len(orig)])

    return vectors


if __name__ == '__main__':
    # ground = [np.array([0, 0, 1, 1]), np.array([0, 0, 0])]
    # preds = [np.array([0, 0, 1, 0]), np.array([0, 0, 1])]

    if len(sys.argv) != 3:
        print("Correct Usage: python  metrics.py  <path_to_input_data_file>  <path_to_output_preds_file>")
        sys.exit(-1)

    ground = read_input_file(sys.argv[1])
    preds = read_preds_file(sys.argv[2], ground)

    results = custom_metrics(ground, preds)
    print("M-Pre: %0.1f; M-Rec: %0.1f;         M-F1: %0.1f; Jacc: %0.1f \n\n"%results)

    