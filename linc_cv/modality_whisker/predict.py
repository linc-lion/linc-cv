

def predict_whisker_path(imagepath):
    X = next(test_datagen.flow(arr, shuffle=False, batch_size=1))
    p = model.predict(X)
    return gt_label, topk_labels, prediction_time


def predict_whisker_url(url):
    return []
