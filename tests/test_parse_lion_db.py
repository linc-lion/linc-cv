import sys
sys.path.append('..')

from linc_cv.parse_lion_db import parse_lion_database

def test_parse_lion_database():
    parse_lion_database(download_db_zip=True)