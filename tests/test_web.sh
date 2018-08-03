#!/bin/sh

# replace ${API_KEY} with an actual API key
# replace ${RESULT_ID} with the job id
# set ${HOST}, e.g. "http://localhost:5000"

curl \
    -H "ApiKey: ${API_KEY}" \
    "${HOST}/linc/v1/capabilities"

curl \
    -H "Content-Type: application/json" \
    -H "ApiKey: ${API_KEY}" \
    --data @test_whisker_classification.json \
    "${HOST}/linc/v1/classify"

curl \
    -H "Content-Type: application/json" \
    -H "ApiKey: ${API_KEY}" \
    --data @test_cv_classification.json \
    "${HOST}/linc/v1/classify"

# curl -H "ApiKey: ${API_KEY}" "${HOST}/linc/v1/results/${RESULT_ID}"

