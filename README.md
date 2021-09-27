# LINC Computer Vision System

## Local setup for Mac

linc-cv uses of 3 components: [Flower](https://flower.readthedocs.io/en/latest/), [Celery](https://docs.celeryproject.org/en/stable/getting-started/introduction.html) and [Supervisor](http://supervisord.org/) 

### linc-cv service setup
* Download [Conda](https://www.anaconda.com/products/individual)
* Run `conda create --name linc-cv python=3.6`
* Run `conda activate linc-cv`
* Run `pip install -r requirements.txt`
* Install [redis](https://gist.github.com/tomysmile/1b8a321e7c58499ef9f9441b2faa0aa8). Celery uses redis message broker.
* Download models from `s3://linc-cv/data/20181223` to `linc_cv/data`

### supervisor setup
* Install [Homebrew](https://brew.sh/)
* Run `brew install supervisor`
* Open `/usr/local/etc/supervisord.conf` with your editor of choice
  * Scroll to the bottom of the page.
  * Replace `files = /usr/local/etc/supervisor.d/*.ini` with `files = /path/to/linc_cv/tests/supervisord/*.conf`.
  * You need to replace `/path/to` with your local path to `linc_cv` project.
* Open `celery.conf` and `flower.conf` under `linc_cv/tests/supervisord`
  * Replace `johndoe` for `command` and `user` variables with your own username. This is the username you use to log in to your machine.
  * You may need to modify the path for `command` if your conda is not installed in the default location.
* Run `sudo /usr/local/opt/supervisor/bin/supervisord -c /usr/local/etc/supervisord.conf --nodaemon`
* `celery-classification.log`, `celery-training.log` and `flower.log` will be created under `linc_cv/tests` folder. 
* Now you should be able to navigate to Flower UI - http://localhost:5555/

### Service startup
* Execute the following code snippet to download the pretrained model:
  * ```python
    > conda activate linc-cv
    > (linc-cv) python
    ```
  * ```python
    >>> model_name = 'senet154'
    >>> model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    ```
  * The pretrained model is saved to `$HOME/.torch`.
* Under project directory `linc-cv`, execute the following in terminal:
  * ```
    > export API_KEY=blah
    > PYTHONPATH=$(pwd) python linc_cv/main.py --web
    ```
    
### Service usage
* Example of request and response for lion face recognition:
  * ```
    curl --location --request POST 'http://192.168.86.137:5000/linc/v1/classify' \
    --header 'ApiKey: blah' \
    --header 'Content-Type: application/json' \
    --data-raw '{
         "type": "cv", 
         "url": "https://raw.githubusercontent.com/linc-lion/linc-cv/lee-dev/tests/images/1311.jpg"
    }'
    ```
    * ```json
     {
      "status": "finished",
      "predictions": [
          {
              "lion_id": "80",
              "probability": 0.412
          },
          {
              "lion_id": "40",
              "probability": 0.032
          },
          {
              "lion_id": "297",
              "probability": 0.028
          },
          ...
       ]
      }
      ```
  * Example of request and response for lion whisker recognition: 
  * ```
    curl --location --request POST 'http://192.168.86.137:5000/linc/v1/classify' \
    --header 'ApiKey: blah' \
    --header 'Content-Type: application/json' \
    --data-raw '{
        "type": "whisker", 
        "url": "https://raw.githubusercontent.com/linc-lion/linc-cv/lee-dev/tests/images/23.jpg"
    }'
    ```
  * ```json
    {
       "status": "finished",
       "predictions": [
           {
               "lion_id": "15",
               "probability": 0.951
           },
           {
               "lion_id": "372",
               "probability": 0.79
           },
           {
               "lion_id": "94",
               "probability": 0.785
           },
           ...
       ]
    }

    ```

## Resources
* [Conda cheat sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)
* [Pretrained models](https://github.com/cadene/pretrained-models.pytorch)
* [Install Supervisor on Mac](https://tn710617.github.io/supervisor/)
* [Setup supervisor on AWS](https://stackoverflow.com/questions/28702780/setting-up-supervisord-on-a-aws-ami-linux-server)