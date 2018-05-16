clear && rsync -rtvz --exclude data --exclude .git \
    --exclude .idea --exclude \*.egg\* --exclude __pycache__  ~/customers/linc/linc-cv-server cassiopeia: