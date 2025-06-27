import os
import pytest
import requests
import json
from pprint import pprint
import time

HOST = os.environ.get('HOST', 'http://localhost:5000')

API_KEY = os.environ.get('API_KEY')
if not API_KEY:
    pytest.fail("API_KEY environment variable not set. Please set it to a valid API key to run integration tests.")

headers = {
    'Content-Type': 'application/json',
    'ApiKey': API_KEY
}

STATUSES_IGNORED = ['PENDING', 'PROGRESS']


def test_capabilities():
    """
    Tests if the /capabilities endpoint is reachable and returns a valid response.
    """ 
    response = requests.get(f'{HOST}/linc/v1/capabilities', headers=headers)
    assert response.status_code == 200, f"Expected status code 200, but got {response.status_code}. Response: {response.text}"

    data = response.json()
    assert 'valid_cv_lion_ids' in data
    assert 'valid_whisker_lion_ids' in data
    assert 'cv_topk_classifier_accuracy' in data
    assert 'whisker_topk_classifier_accuracy' in data


def _run_classification_test(test_file):
    test_file_path = os.path.join(os.path.dirname(__file__), test_file)
    with open(test_file_path, 'r') as f:
        data = json.load(f)

    # 1. Submit the classification job
    response = requests.post(f'{HOST}/linc/v1/classify', json=data, headers=headers)
    assert response.status_code == 200, f"Failed to submit job. Status: {response.status_code}, Response: {response.text}"

    result = response.json()
    result_id = result['id']

    # 2. Poll for the result
    start_time = time.time()
    timeout = 60  # seconds
    final_result = None

    while time.time() - start_time < timeout:
        time.sleep(2)

        result_response = requests.get(f'{HOST}/linc/v1/results/{result_id}', headers=headers)
        assert result_response.status_code == 200, f"Failed to get result. Status: {result_response.status_code}, Response: {result_response.text}"

        final_result = result_response.json()
        if final_result.get('status') not in STATUSES_IGNORED:
            break
    else:
        pytest.fail(f"Test timed out after {timeout} seconds waiting for result for job {result_id}")

    print()
    pprint(final_result)
    assert final_result.get('status') == 'finished'

    return final_result


def test_cv_classification():
    """
    Tests the /classify endpoint using CV classification test data.
    """
    final_result = _run_classification_test('test_cv_classification.json')
    
    # Add any CV-specific assertions here
    assert len(final_result['predictions']) > 0


def test_whisker_classification():
    """
    Tests the /classify endpoint using whisker classification test data.
    """
    final_result = _run_classification_test('test_whisker_classification.json')
    
    # Add any whisker-specific assertions here
    assert len(final_result['predictions']) > 0
