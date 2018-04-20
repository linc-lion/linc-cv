#!/bin/sh

celery -A tasks worker -E --concurrency=1 --max-tasks-per-child=1
