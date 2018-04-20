import requests


def scrape_lion_idx(idx):
    j = None
    try:
        j = requests.get('https://linc-api.herokuapp.com/lions/' + str(idx)).json()
        print('successfully scraped database id -> ' + str(idx))
    except:
        print('failed to scrape database id -> ' + str(idx))
    return j
