#!/bin/sh

clear && rsync -rtvz --delete --exclude data --exclude .git \
    --exclude .pytest* --exclude .idea --exclude \*.egg\* --exclude __pycache__  \
    ~/customers/linc/linc-cv-server/ justin-linc:linc-cv-server
