import time
import json
import os
from operator import itemgetter

import numpy as np
from PIL import Image
from keras.models import load_model
from toolz.itertoolz import partition_all
from keras.preprocessing.image import ImageDataGenerator, img_to_array


with open('class_indicies.json') as f:
    class_indicies = json.load(f)
model = load_model('whiskers.h5')
labels = [x[0] for x in sorted(class_indicies.items(), key=itemgetter(1))]
num_classes = len(labels)
print(f'num_classes -> {num_classes}')
test_datagen = ImageDataGenerator(
    rescale=1. / 255,
    samplewise_center=True,
    samplewise_std_normalization=True,)


def process(path):
    start_time = time.time()
    im = Image.open(path).convert('RGB')
    im = im.resize((299, 299,))
    im = img_to_array(im)
    im = np.expand_dims(im, 0)
    assert im.shape == (1, 299, 299, 3,), im.shape
    gt_label = path.split('/')[-2]

    y = np.zeros((1, num_classes,))
    y[0][class_indicies[gt_label]] = 1

    X = next(test_datagen.flow(im, shuffle=False, batch_size=1))
    p = model.predict(X)

    pred_label = labels[np.argmax(p, axis=1)[0]]
    correct = pred_label == gt_label
    total_time = time.time() - start_time
    return gt_label, pred_label, correct, total_time


if __name__ == '__main__':
    from collections import defaultdict
    classifications = defaultdict(list)
    all_times = []
    all_scores = []
    for root, dirs, files in os.walk('whiskers_traintest/test'):
        for f in files:
            path = os.path.join(root, f)
            gt_label, pred_label, correct, total_time = process(path)
            all_scores.append(correct)
            all_times.append(total_time)
            classifications[gt_label].append(pred_label)

    for gt_label, pred_labels in classifications.items():
        scores = []
        for pred_label in pred_labels:
            scores.append(pred_label == gt_label)
        print(f'label: {gt_label}, accuracy: {np.around(np.mean(scores), 3)}')
    print(f'OVERALL, number of test samples: {len(all_scores)}, '
          f'accuracy: {np.around(np.mean(all_scores), 3)}')
    print(f'OVERALL, mean time to perform one '
          f'prediction: {np.around(np.mean(all_times), 3)}')
