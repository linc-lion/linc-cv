#!/bin/sh

# replace ${API_KEY} with an actual API key
# replace ${RESULT_ID} with the job id

curl \
    -H "ApiKey: ${API_KEY}" \
    http://localhost:5000/linc/v1/whisker/capabilities

curl \
    -H "Content-Type: application/json" \
    -H "ApiKey: ${API_KEY}" \
    --data @test_whisker_classification.json \
    http://localhost:5000/linc/v1/classify

curl \
    -H "Content-Type: application/json" \
    -H "ApiKey: ${API_KEY}" \
    --data @test_lion_classification.json \
    http://localhost:5000/linc/v1/classify

curl -H "ApiKey: ${API_KEY}" http://localhost:5000/linc/v1/results/${RESULT_ID}

