import os
from collections import Counter
from sklearn.model_selection import train_test_split
import shutil


def process(xs, ys, mode):
    for xt, yt in zip(xs, ys):
        _, label, f = xt.split('/')
        src = xt
        dst = f'whiskers_traintest/{mode}/{label}/{f}'
        print(src, dst)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)


if __name__ == '__main__':

    try:
        shutil.rmtree('whiskers_traintest')
    except FileNotFoundError:
        pass

    X = []
    y = []

    for root, dirs, files in os.walk('whiskers_filtered'):
        for f in files:
            path = os.path.join(root, f)
            label = path.split('/')[-2]
            X.append(path)
            y.append(label)

    c = Counter(y)
    min_samples_per_label = 2
    ok_ys = set(y for y in c if c[y] > min_samples_per_label)

    Xf = []
    yf = []
    for xt, yt in zip(X, y):
        if yt in ok_ys:
            Xf.append(xt)
            yf.append(yt)

    X_train, X_test, y_train, y_test = train_test_split(Xf, yf, stratify=yf)
    process(X_train, y_train, mode='train')
    process(X_test, y_test, mode='test')
