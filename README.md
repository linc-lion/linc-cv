# LINC Computer Vision System

## Local setup for Mac

linc-cv uses of 3 components: [Flower](https://flower.readthedocs.io/en/latest/), [Celery](https://docs.celeryproject.org/en/stable/getting-started/introduction.html) and [Supervisor](http://supervisord.org/) 

### linc-cv service setup
* Download [Conda](https://www.anaconda.com/products/individual)
* Run `conda create --name linc-cv python=3.6`
* Run `conda activate linc-cv`
* Run `pip install -r requirements.txt`
* Install [redis](https://gist.github.com/tomysmile/1b8a321e7c58499ef9f9441b2faa0aa8). Celery uses redis message broker.

### supervisor setup
* Install [Homebrew](https://brew.sh/)
* Run `brew install supervisor`
* Open `/usr/local/etc/supervisord.conf` with your editor of choice
  * Scroll to the bottom of the page.
  * Replace `files = /usr/local/etc/supervisor.d/*.ini` with `files = /path/to/linc-cv/tests/supervisord/*.conf`.
  * You need to replace `/path/to` with your local path to `linc_cv` project.
* Open `celery.conf` and `celery.conf` under `linc_cv/tests/supervisord`
  * Replace `johndoe` for `command` and `user` variables with your own username. This is the username you use to log in to your machine.
* Run `sudo /usr/local/opt/supervisor/bin/supervisord -c /usr/local/etc/supervisord.conf --nodaemon`
* `celery-classification.log`, `celery-training.log` and `flower.log` will be created under `linc_cv/tests` folder. 
