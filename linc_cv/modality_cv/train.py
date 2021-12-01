import os
import shutil
from collections import Counter
import json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pretrainedmodels
import pretrainedmodels.utils as utils
import torch.nn
import numpy as np
from tqdm import tqdm
import joblib
from sklearn.metrics import classification_report

from linc_cv.settings import CV_IMAGES_PATH, CV_IMAGES_TRAINTEST_PATH, \
    CV_CLASSIFIER_PATH, CV_FEATURES_TRAIN_X, CV_FEATURES_TRAIN_Y, \
    CV_FEATURES_TEST_X, CV_FEATURES_TEST_Y, CV_MODEL_CLASSES_JSON


def extract_features_cv(*, images_dir, images_traintest_dir):
    try:
        shutil.rmtree(images_traintest_dir)
    except FileNotFoundError:
        pass

    X = []
    y = []
    for root, dirs, files in os.walk(images_dir):
        for f in files:
            path = os.path.join(root, f)
            label = path.split(os.path.sep)[-2]
            X.append(path)
            y.append(label)
    valid_labels = set(label for label, count in Counter(y).items() if count > 2)

    Xp = []
    yp = []
    for x_, y_ in zip(X, y):
        if y_ in valid_labels:
            Xp.append(x_)
            yp.append(y_)

    X_train, X_test, y_train, y_test = train_test_split(Xp, yp, stratify=yp, shuffle=True)

    for x, y, mode in ((X_train, y_train, 'train',), (X_test, y_test, 'test',),):
        for a, b in zip(x, y):
            np = os.path.join(images_traintest_dir, mode, b, os.path.basename(a) + '.jpg')
            os.makedirs(os.path.dirname(np), exist_ok=True)
            os.symlink(a, np)
    extract_features(
        x_path=CV_FEATURES_TRAIN_X, y_path=CV_FEATURES_TRAIN_Y,
        rootdir=os.path.join(images_traintest_dir, 'train'), mode='train')

    extract_features(
        x_path=CV_FEATURES_TEST_X, y_path=CV_FEATURES_TEST_Y,
        rootdir=os.path.join(images_traintest_dir, 'test'), mode='test')


class CV_NN_Model(object):
    def __init__(self, use_cuda=True):
        self.use_cuda = use_cuda
        print('loading pytorch model')
        self.ap2d = torch.nn.AvgPool2d(7)
        model_name = 'senet154'  # could be fbresnet152 or inceptionresnetv2
        model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        model.eval()
        if use_cuda:
            model.cuda()
        self.model = model
        self.load_img = utils.LoadImage()
        # transformations depending on the model
        # rescale, center crop, normalize, and others (ex: ToBGR, ToRange255)
        self.tf_img = utils.TransformImage(model)
        print('loaded pytorch model')

    def predict(self, path_img):
        input_img = self.load_img(path_img)
        input_tensor = self.tf_img(input_img)  # 3x400x225 -> 3x299x299 size may differ
        input_tensor = input_tensor.unsqueeze(0)  # 3x299x299 -> 1x3x299x299
        input_ = torch.autograd.Variable(input_tensor, requires_grad=False)
        if self.use_cuda:
            output_features = self.model.features(input_.cuda())  # 1x14x14x2048 size may di
        else:
            output_features = self.model.features(input_)  # 1x14x14x2048 size may differ
        output_features_f = self.ap2d(output_features)
        if self.use_cuda:
            output_features_f = output_features_f.cpu()
        output_features_f = output_features_f.flatten().detach().numpy()
        return output_features_f


def extract_features(x_path, y_path, rootdir, mode):
    cv_nn_model = CV_NN_Model(use_cuda=False)
    ps = []
    for root, dirs, files in os.walk(rootdir):
        for f in files:
            p = os.path.join(root, f)
            ps.append(p)
    shape = (len(ps), 2048,)
    n = np.zeros(shape, dtype='float32', )
    labels = []
    for idx, path_img in enumerate(tqdm(ps, desc=mode)):
        n[idx] = cv_nn_model.predict(path_img)
        label = path_img.split(os.sep)[-2]
        labels.append(label)
    np.save(x_path, n)
    with open(y_path, 'w') as fd:
        json.dump(labels, fd)


def train(x_train_path, y_train_path, x_test_path, y_test_path, clf_save_path):
    X_train = np.load(x_train_path)
    with open(y_train_path) as fd:
        y_train = json.load(fd)
    X_test = np.load(x_test_path)
    with open(y_test_path) as fd:
        y_test = json.load(fd)
    clf = RandomForestClassifier(
        n_estimators=500, class_weight='balanced', oob_score=True, n_jobs=-1, verbose=2)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(f'clf score: {round(score, 3)}')
    print(classification_report(y_test, clf.predict(X_test)))
    joblib.dump(clf, clf_save_path, compress=('xz', 9,))
    with open(CV_MODEL_CLASSES_JSON, 'w') as fd:
        json.dump(clf.classes_.tolist(), fd)


def extract_cv_features():
    return extract_features_cv(
        images_dir=CV_IMAGES_PATH,
        images_traintest_dir=CV_IMAGES_TRAINTEST_PATH, )


def train_cv_classifier():
    return train(
        x_train_path=CV_FEATURES_TRAIN_X, y_train_path=CV_FEATURES_TRAIN_Y,
        x_test_path=CV_FEATURES_TEST_X, y_test_path=CV_FEATURES_TEST_Y,
        clf_save_path=CV_CLASSIFIER_PATH)
