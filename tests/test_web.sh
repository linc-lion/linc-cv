#!/bin/sh

curl -H "Content-Type: application/json" --data @test_whisker_classification.json \
    http://localhost:5000/linc/v1/classify

curl -H "Content-Type: application/json" --data @test_lion_classification.json \
    http://localhost:5000/linc/v1/classify

# curl http://localhost:5000/linc/v1/results/<<< result id >>>

