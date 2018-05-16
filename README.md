# LINC Computer Vision System

## Installation

    ./install_dev.sh
    linc_cv --scrape-lion-database
    linc_cv --generate-images-lut
    linc_cv --train-cv
    linc_cv --validate-cv
    linc_cv --train-whiskers
    linc_cv --validate-whiskers

## Runtime

### Launch web worker

    linc_cv --web

### Launch web worker

    linc_cv --worker

### Launch web worker monitor

    linc_cv --flower
