# LINC Computer Vision System

## Installation

    ./install_dev.sh
    linc_cv --scrape-lion-database
    linc_cv --generate-images-lut

    linc_cv --download-cv-images
    linc_cv --train-cv-classifier
    linc_cv --cv-classifier-report
    linc_cv --validate-cv-classifier

    linc_cv --download-whisker-images
    linc_cv --train-whisker-classifier
    linc_cv --whisker-classifier-report
    linc_cv --validate-whisker-classifier

## Runtime

### Launch web worker

    linc_cv --web

### Launch web worker

    linc_cv --worker

### Launch web worker monitor

    linc_cv --flower
