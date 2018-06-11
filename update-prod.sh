#!/bin/sh

rsync -rtvz --progress --stats ~/customers/linc/linc-cv-server/*/*.py linc_testing:linc-cv-server
