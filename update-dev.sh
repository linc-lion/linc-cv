#!/bin/sh

clear && rsync -rtvz --exclude data --exclude .git \
    --exclude .pytest* --exclude .idea --exclude \*.egg\* --exclude __pycache__  \
    . cassiopeia:linc-cv

