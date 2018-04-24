rsync -rtv --stats \
    --exclude "linc_cv/data/" \
    --exclude ".git" \
    --exclude ".DS_Store" \
    --exclude ".idea" \
    ~/customers/linc/linc-cv-server cassiopeia-lan:
