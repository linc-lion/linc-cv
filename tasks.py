from test_classify import test_lion, ClassifierError
from celery import Celery

c = Celery()
c.conf.broker_url = 'redis://localhost:6379/0'
c.conf.result_backend = 'redis://localhost:6379/0'


@c.task()
def classify_image_url_against_lion_ids(test_image_url, feature_type, lion_ids):
    try:
        feature_type, correct, val_acc, probas, labels = test_lion(
            feature_type=feature_type, lion_ids=lion_ids, test_image_url=test_image_url)

        results = []
        for lion_id, probability in zip(labels, probas[0]):
            results.append(
                {'classifier': round(float(val_acc), 3),
                 'confidence': round(float(probability), 3),
                 'id': int(lion_id)})
        return results

    except ClassifierError as e:
        return {
            'status': 'error',
            'info': e.message}

    return {
        'status': 'finished',
        'match_probability': task_result}


if __name__ == '__main__':
    test_image_url = 'http://pixdaus.com/files/items/pics/7/84/542784_81cef138c75698faddafee92d42c0cc5_large.jpg'
    feature_type = 'main-id'
    lion_ids = ['131', '234', '142', '97']
    result = classify_image_url_against_lion_ids(test_image_url, feature_type, lion_ids)
    print(result)
