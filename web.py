from flask import request
from flask import Flask
from flask_restful import Resource, Api

from classify import classify_image_url_against_lion_ids
from classify import c


class LincResultAPI(Resource):
    def get(self, celery_id):
        """
        {
          identification: {
            id: "e3464fc5-54d5-48db-bc05-761cad999cd8",
            status: "finished",
            lions: [                           #  Each lion object in the array will include an id and confidence
                                                         # (in the range 0-1.0)
              {id: 35, confidence: 0.8},
              {id: 23, confidence: 0.6}
            ]
          }
        }
        """

        my_task = c.AsyncResult(id=celery_id)
        try:
            if my_task.ready():
                task_result = my_task.get()
                if task_result is None:
                    return {'status': 'error', 'info': 'failed to process input image'}
                return {
                    'status': 'finished',
                    'match_probability': task_result
                }
            else:
                return {
                    'status': 'pending',
                    'match_probability': []
                }
        except:
            import traceback
            traceback.print_exc()
            traceback.print_stack()
            return {'status': 'error'}, 400


class LincClassifyAPI(Resource):
    def post(self):
        """
        {
            "identification": {
                "images": [
                    {
                        "id": 123,
                        "type": "whisker",
                        "url": "https://s3.amazonaws.com/semanticmd-api-testing/api/cbc90b5705d51e9e218b0a7e518aa6d3506c1"
                    }
                ],
            "gender": "m",
            "age": 5,
            "lions": [
                {
                    "id": 456,
                    "url": "http://lg-api.com/lions/456",
                    "updated_at": "timestamp"
                }
            ]
            }
        }
        """

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
                lions_ids = [r['id'] for r in json_request['identification']['lions']]
            except (TypeError, AttributeError,):
                return {'status': 'error', 'info': 'could not parse lion ids from identifications'}, 400

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
    app.run(debug=True)
