from linc_cv.modality_whisker.predict import predict_whisker_url

url = 'http://livingwithlions.org/mara/images/lion_whiskers_image-134.jpg'


def test_whisker():
    print(predict_whisker_url(url))


if __name__ == '__main__':
    test_whisker()
