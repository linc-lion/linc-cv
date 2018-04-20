#!/bin/sh

celery -A linc_cv.tasks worker -E --concurrency=1 --max-tasks-per-child=1
