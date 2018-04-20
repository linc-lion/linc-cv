import json
import multiprocessing
import requests


def process(idx):
    j = None
    try:
        j = requests.get('https://linc-api.herokuapp.com/lions/' + str(idx)).json()
        print('success -> ' + str(idx))
    except:
        print('failed -> ' + str(idx))
    return j


if __name__ == '__main__':
    data = []
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        for result in pool.imap_unordered(process, list(range(1500))):
            data.append(result)

    with open('data/linc_db.json', 'w') as f:
        json.dump(data, f)
