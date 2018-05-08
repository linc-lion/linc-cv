# coding=utf-8

import os
import shutil
from collections import Counter

from sklearn.model_selection import train_test_split

from linc_cv import datapath


def process(xs, ys, mode):
    for xt, yt in zip(xs, ys):
        *base, label, f = xt.split('/')
        src = xt
        dst = datapath(['whiskers_images_traintest', f'{mode}/{label}/{f}'])
        print(src, dst)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)


def whiskers_train_test_split():
    """
    Convert preprocessed lion images into to a training set and a validation set
    for neural network training
    """
    try:
        shutil.rmtree(datapath(['whiskers_images_traintest']))
    except FileNotFoundError:
        pass

    X = []
    y = []

    for root, dirs, files in os.walk(datapath(['whiskers_images_normalized'])):
        for f in files:
            path = os.path.join(root, f)
            label = path.split('/')[-2]
            X.append(path)
            y.append(label)

    c = Counter(y)

    # only include lion_ids with a minimum number of whisker images
    ok_ys = set(y for y in c if c[y] > 2)

    Xf = []
    yf = []
    for xt, yt in zip(X, y):
        if yt in ok_ys:
            Xf.append(xt)
            yf.append(yt)

    X_train, X_test, y_train, y_test = train_test_split(Xf, yf, stratify=yf)
    process(X_train, y_train, mode='train')
    process(X_test, y_test, mode='test')
