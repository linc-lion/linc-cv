# LINC Lion Identification Service

A lion identification service based on lion face and whiskers recognitions.

## Deployment
This application is currently deployed via a blue green methodology using Github Actions. The process is as follows:
1. Work on code changes in a feature branch based off of the staging branch
2. Submit a merge request to the staging branch
3. Run the ***Deploy*** Github Action workflow pointed to the staging branch
4. Receive approval from project administrators
5. Submit a merge request to the master branch
6. Determine which environment (blue, green) is active in production
7. Run the ***Deploy*** Github Action workflow pointed to the inactive environment
8. Point the staging and production webapp in Heroku to the inactive environment, making it active
9. Run the ***Destroy*** Github Action workflow pointed to the new inactive environment

## linc-cv training
* Clone [linc-cv-data](https://github.com/linc-lion/linc-cv-data).
* Create a `data` folder under linc-cv/linc-cv.
* Copy `whisker_model_yolo.h5` from `linc-cv-data` to linc-cv/linc-cv/data.
  * The `whisker_model_yolo.h5` model was built by previous developers. Unfortunately, the training code is missing.
* Export the following ENV variables:
  * LINC_USERNAME
  * LINC_PASSWORD
* Execute the following training commands in linc-cv/linc-cv/main.py:
  * python <path_to>/linc-cv/linc_cv/main.py --parse-lion-database
  * python <path_to>/linc-cv/linc_cv/main.py --download-cv-images
  * python <path_to>/linc-cv/linc_cv/main.py --extract-cv-features
  * python <path_to>/linc-cv/linc_cv/main.py --train-cv-classifier
  * python <path_to>/linc-cv/linc_cv/main.py --download-whisker-images
  * python <path_to>/linc-cv/linc_cv/main.py --train-whisker-classifier

## Local setup for Mac

linc-cv uses 3 components: [Flower](https://flower.readthedocs.io/en/latest/), [Celery](https://docs.celeryproject.org/en/stable/getting-started/introduction.html) and [Supervisor](http://supervisord.org/)

### linc-cv service setup
* Download [Conda](https://www.anaconda.com/products/individual)
* Run `conda create --name linc-cv python=3.6`
* Run `conda activate linc-cv`
* Run `pip install -r requirements.txt`
* Install [redis](https://gist.github.com/tomysmile/1b8a321e7c58499ef9f9441b2faa0aa8). Celery uses redis message broker.
* Download models from [linc-cv-data repository](https://github.com/linc-lion/linc-cv-data) to `linc_cv/data`

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
    >>> import pretrainedmodels
    >>> model_name = 'senet154'
    >>> model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    ```
  * The pretrained model is saved to `$HOME/.torch`.
* Under project directory `linc-cv`, execute the following in terminal:
  * ```
    > export API_KEY=blah
    > PYTHONPATH=$(pwd) python linc_cv/web.py
    ```
    
### Service usage
* Example of request and response (truncated for brievity) for lion face recognition:
  * ```
    curl --location --request POST 'http://192.168.86.137:5000/linc/v1/classify' \
    --header 'ApiKey: blah' \
    --header 'Content-Type: application/json' \
    --data-raw '{
         "type": "cv", 
         "url": "https://raw.githubusercontent.com/linc-lion/linc-cv/master/tests/images/female_lion_face_1.jpeg"
    }'
    ```
  * ```json
    {
       "id": "f9591d42-96e6-4178-9022-cab02cd86b3b",
       "status": "PENDING",
       "errors": []
    }
    ```
  * ```
    curl --location --request GET 'http://192.168.86.137:5000//linc/v1/results/f9591d42-96e6-4178-9022-cab02cd86b3b' \
    --header 'ApiKey: blah' \
    --header 'Content-Type: application/json'
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
           }
       ]
    }
    ```
  * Example of request and response (truncated for brievity) for lion whisker recognition: 
  * ```
    curl --location --request POST 'http://192.168.86.137:5000/linc/v1/classify' \
    --header 'ApiKey: blah' \
    --header 'Content-Type: application/json' \
    --data-raw '{
        "type": "whisker", 
        "url": "https://raw.githubusercontent.com/linc-lion/linc-cv/master/tests/images/sample_lion_whisker_23.jpg"
    }'
    ```
  * ```json
    {
       "id": "3f6dbfdf-98ea-4d76-92af-e5ff9912546b",
       "status": "PENDING",
       "errors": []
    }
    ```
  * ```
    curl --location --request GET 'http://192.168.86.137:5000//linc/v1/results/3f6dbfdf-98ea-4d76-92af-e5ff9912546b' \
    --header 'ApiKey: blah' \
    --header 'Content-Type: application/json'
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
           }
       ]
    }

    ```


## Resources
* [Conda cheat sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)
* [Pretrained models](https://github.com/cadene/pretrained-models.pytorch)
* [Install Supervisor on Mac](https://tn710617.github.io/supervisor/)
* [Setup supervisor on AWS](https://stackoverflow.com/questions/28702780/setting-up-supervisord-on-a-aws-ami-linux-server)