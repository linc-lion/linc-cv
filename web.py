from flask import Flask
from flask import request
from flask_restful import Resource, Api

from tasks import c, classify_image_url_against_lion_ids


class LincResultAPI(Resource):
    def get(self, celery_id):
        t = c.AsyncResult(id=celery_id)
        if t.ready():
            return t.get()
        return {
            'status': 'pending',
            'match_probability': []
        }


class LincClassifyAPI(Resource):
    def post(self):
        try:
            json_request = request.get_json()
            if json_request is None:
                return {'status': 'error', 'info': 'could not parse JSON data from request'}, 400
            try:
                image_type = json_request['identification']['images'][0]['type']
            except KeyError:
                return {'status': 'error', 'info': 'missing image type in identification'}, 400
            try:
                image_url = json_request['identification']['images'][0]['url']
            except KeyError:
                return {'status': 'error', 'info': 'missing url in image identification'}, 400
            try:
                lions_ids = [str(r['id']) for r in json_request['identification']['lions']]
            except (TypeError, AttributeError,):
                return {'status': 'error', 'info': 'could not parse lion ids from identifications'}, 400

            print(image_url, image_type, lions_ids)
            job_id = classify_image_url_against_lion_ids.delay(image_url, image_type, lions_ids).id

            return {
                'id': job_id,
                'status': 'pending',
                'lions': []
            }
        except KeyError:
            return {'status': 'error'}, 400


app = Flask(__name__)
api = Api(app)

api.add_resource(LincClassifyAPI, '/linc/v1/classify')
api.add_resource(LincResultAPI, '/linc/v1/results/<string:celery_id>')

if __name__ == '__main__':
    if __debug__:
        app.run(debug=True)
    else:
        app.run(debug=False, host='0.0.0.0')
