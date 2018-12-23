from io import BytesIO
from redis import StrictRedis
import requests
from operator import itemgetter

from sklearn.metrics.pairwise import pairwise_distances
from PIL import Image
import cv2
import numpy as np
import joblib

from linc_cv import WHISKER_FEATURE_X_PATH, WHISKER_FEATURE_Y_PATH, \
    WHISKER_BBOX_MODEL_PATH, REDIS_MODEL_RELOAD_KEY, ClassifierError
from .icp import icp
from .inference import YOLO

whisker_bbox_model = None
X = None
y = None


def resize_to_longer_edge(im, target_size):
    """
    Make image square by enlarging it, then resize to target_size
    Input and output: PIL Image
    """

    width, height = im.size
    if width > height:
        im_n = Image.new('L', (width, width,))
        offset = (width - height) // 2
        im_n.paste(im, (0, offset,))
    elif height > width:
        im_n = Image.new('L', (height, height,))
        offset = (height - width) // 2
        im_n.paste(im, (offset, 0,))
    else:
        im_n = im
    return im_n.resize(target_size)


def simplify_whisker(im, d, e, ma, t1, t2):
    clahe = cv2.createCLAHE()
    im = clahe.apply(im)
    _, t = cv2.threshold(im, t1, t2, cv2.THRESH_BINARY)

    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3,))
    t = cv2.dilate(t, kernel3, iterations=d)
    t = cv2.erode(t, kernel3, iterations=e)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByInertia = False
    params.filterByConvexity = False
    params.filterByArea = True
    params.minArea = ma
    d = cv2.SimpleBlobDetector_create(params)
    keypoints = d.detect(t)

    return None, keypoints


def comparator(A, B):
    if A.size == 0 or B.size == 0:
        return None

    if A.shape != B.shape:
        discrepancy = abs(A.shape[0] - B.shape[0])
        dists = pairwise_distances(A, B)
        idxs = np.argsort(-dists.ravel())
        pts_a, pts_b = np.unravel_index(idxs, dists.shape)
        if len(A) > len(B):
            A_m = np.ones(len(A), dtype=np.bool)
            idxs = set()
            pts_a = iter(pts_a)
            while len(idxs) < discrepancy:
                idxs.add(next(pts_a))
            A_m[list(idxs)] = False
            A = A[A_m]
        else:
            B_m = np.ones(len(B), dtype=np.bool)
            idxs = set()
            pts_b = iter(pts_b)
            while len(idxs) < discrepancy:
                idxs.add(next(pts_b))
            B_m[list(idxs)] = False
            B = B[B_m]

    distances = icp(A, B)

    # median is best so far... 46% @ top-10
    return np.median(distances)


def preprocess(image, bbox, d, e, ma, t1, t2, sz):
    crop = image.convert('L')
    crop = crop.crop(bbox)
    crop = resize_to_longer_edge(crop, target_size=(sz, sz,))
    crop = np.array(crop, dtype=np.uint8)
    im, kpts = simplify_whisker(crop, d, e, ma, t1, t2)
    feature = np.array([k.pt for k in kpts], dtype=np.float64)
    return feature


def predict_whisker_url(test_image_url):
    global whisker_bbox_model
    global X
    global y
    if X is None or y is None:
        X = joblib.load(WHISKER_FEATURE_X_PATH)
        y = joblib.load(WHISKER_FEATURE_Y_PATH)

    if whisker_bbox_model is None:
        whisker_bbox_model = YOLO(WHISKER_BBOX_MODEL_PATH)

    r = requests.get(test_image_url)
    if not r.ok:
        return None
    buf = BytesIO(r.content)
    try:
        image = Image.open(buf).convert('RGB')
    except OSError:
        return None
    rois = whisker_bbox_model.detect_image(image)
    if not rois:
        raise ClassifierError('No whisker ROIs found in source image.')

    # select roi with highest detection probability
    roi = sorted(rois, key=itemgetter(1), reverse=True)[0]
    _, confidence, im_x, im_y, im_w, im_h = roi

    # magic values discovered using prior probabilistic regression experiments
    # TODO: move these magic values into a common dictionary
    d = 5
    e = 2
    ma = 15
    t1 = 53
    t2 = 120
    sz = 400
    topk = 10

    if confidence < 0.98:
        raise ClassifierError(f'Insufficient ROI confidence score. Ignoring detected ROI. Confidence: {confidence}')
    bbox = (im_y, im_x, im_h, im_w,)

    A = preprocess(image, bbox, d, e, ma, t1, t2, sz)
    whisker_scores = []
    for idx, B in enumerate(X):
        score = comparator(A, B)
        whisker_scores.append([idx, score])
    whisker_scores = sorted(whisker_scores, key=itemgetter(1))
    max_score = max(whisker_scores, key=itemgetter(1))[1]
    results = {'predictions': []}
    for idx, score in whisker_scores[:topk]:
        lion_id = y[idx]
        probability = round(1 - score / max_score, 3)
        results['predictions'].append({'lion_id': lion_id, 'probability': probability})
    return results
