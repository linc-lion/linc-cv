#!/bin/sh

curl -H "Content-Type: application/json" --data @test_web.json \
    http://localhost:5000/linc/v1/classify

curl http://localhost:5000/linc/v1/results/cd538def-7dd2-4efb-8470-71312995f4b0