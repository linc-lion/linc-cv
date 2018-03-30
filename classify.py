import json
import tempfile
import pickle
import shutil
from typing import Iterable
import multiprocessing

import requests
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing import image
from keras.models import Model

import numpy as np
from celery import Celery

c = Celery()
FEATURE_SHAPE = (512,)


def init_process():
    global model
    base_model = VGG19(weights='imagenet', include_top=False)
    bmo = base_model.output
    bmo = GlobalAveragePooling2D()(bmo)
    model = Model(inputs=base_model.input, outputs=bmo)


def extract_image_url_features(image_url):
    with tempfile.NamedTemporaryFile() as t:
        r = requests.get(image_url, stream=True)
        shutil.copyfileobj(r.raw, t)
        t.flush()
        t.seek(0)
        try:
            img = image.load_img(t.name, target_size=(224, 224))
        except OSError:
            return None
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features


def generate_lion_db_singleprocess():
    with open('linc_images_db.json') as f:
        linc_images_db = json.load(f)
    image_count = 0
    for lion_id, feature_types in linc_images_db.items():
        for feature_type, image_urls in feature_types.items():
            image_count += len(image_urls)
    lion_features_arr = np.memmap('lion_features.npy', dtype='float32', mode='w+', shape=(image_count,) + FEATURE_SHAPE)
    lion_feature_db = {}
    lion_features_image_index = 0
    for lion_id, feature_types in linc_images_db.items():
        for feature_type, image_urls in feature_types.items():
            for image_url in image_urls:
                image_features = extract_image_url_features(image_url)
                lion_features_arr[lion_features_image_index] = image_features
                lion_feature_db.setdefault(lion_id, {})
                lion_feature_db[lion_id].setdefault(feature_type, [])
                lion_feature_db[lion_id][feature_type].append(lion_features_image_index)
                print(f'generated and saved features for image url # {lion_features_image_index} of {image_count}')
                lion_features_image_index += 1
    with open('lion_feature_db.pkl', 'wb') as f:
        pickle.dump(lion_feature_db, f)
    return lion_feature_db


def process_urls(data):
    lion_features_image_index, image_url = data
    features = extract_image_url_features(image_url)
    return lion_features_image_index, features


def generate_lion_db():
    with open('linc_images_db.json') as f:
        linc_images_db = json.load(f)
    image_count = 0
    for lion_id, feature_types in linc_images_db.items():
        for feature_type, image_urls in feature_types.items():
            image_count += len(image_urls)
    lion_feature_db = {}
    lion_features_image_index = 0
    data = []
    bad_ids = []
    for lion_id, feature_types in linc_images_db.items():
        for feature_type, image_urls in feature_types.items():
            for image_url in image_urls:
                data.append((lion_features_image_index, image_url,))
                lion_feature_db.setdefault(lion_id, {})
                lion_feature_db[lion_id].setdefault(feature_type, [])
                lion_feature_db[lion_id][feature_type].append(lion_features_image_index)
                lion_features_image_index += 1
    lion_features_arr = np.memmap(
        'lion_features.npy', dtype='float32', mode='w+', shape=(image_count,) + FEATURE_SHAPE)
    with multiprocessing.Pool(
            initializer=init_process,
            processes=multiprocessing.cpu_count() // 2) as pool:
        for lion_features_image_index, image_features in pool.imap_unordered(process_urls, data):
            if image_features is None:
                bad_ids.append(lion_features_image_index)
                print(f'skipped image url # {lion_features_image_index} of {image_count}')
            else:
                print(f'generated and saved features for image url # {lion_features_image_index} of {image_count}')
                lion_features_arr[lion_features_image_index] = image_features
    with open('lion_feature_db.pkl', 'wb') as f:
        pickle.dump(lion_feature_db, f)
    return lion_feature_db


try:
    with open('lion_feature_db.pkl', 'rb') as f:
        lion_feature_db = pickle.load(f)
except FileNotFoundError:
    lion_feature_db = generate_lion_db()


@c.task()
def classify_image_url_against_lion_ids(
        image_url: str, feature_type: str, lion_ids: Iterable):
    feature = extract_image_url_features(image_url)
    X = []
    y = []
    for lion_id, lion_feature in lion_feature_db:
        if lion_id in lion_ids:
            X.append(lion_feature)
            y.append(lion_id)
    try:
        X = np.concatenate(X)
        y = np.array(y)
    except ValueError:  # no such lion in the database, for example
        return []
    return classify(feature, X, y)


def classify(feature, X, y):
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import StratifiedKFold

    # evaluate the predictive power of the classifier
    scores = []
    classes, counts = np.unique(y, return_counts=True)
    n_splits = np.amin(counts)
    if n_splits > 10:
        n_splits = 10
    if len(y) < 2:
        return {'status': 'error', 'info': 'result search space too small'}
    print('n_splits: ' + str(n_splits))
    print('counts: ' + str(counts.tolist()))
    for train_idx, test_idx in StratifiedKFold(
            n_splits=n_splits).split(X, y):
        clf = MLPClassifier(
            hidden_layer_sizes=(512, 512,), early_stopping=True, tol=1e-5,
            verbose=True, max_iter=3000)
        clf.fit(X[train_idx], y[train_idx])
        score = clf.score(X[test_idx], y[test_idx])
        scores.append(score)
    confidence = np.mean(scores)
    clf = MLPClassifier(
        hidden_layer_sizes=(512, 512,), early_stopping=True, tol=1e-5,
        verbose=True, max_iter=3000)
    clf.fit(X, y)
    try:
        probabilities = clf.predict_proba(feature)[0]
    except ValueError:
        # ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
        print('encountered a value error in probability predictor')
        return []
    results = []
    for i, lion_id in enumerate(clf.classes_):
        prob = probabilities[i]
        results.append(
            {'classifier': round(float(confidence), 3),
             'confidence': round(float(prob), 3),
             'id': int(lion_id)})
    return results


if __name__ == '__main__':
    lion_feature_db = generate_lion_db()
