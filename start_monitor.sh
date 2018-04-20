#!/bin/sh

celery flower -A linc_cv.tasks -E
