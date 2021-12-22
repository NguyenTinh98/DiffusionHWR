from flask import Flask,url_for,render_template,request
import json
import torch
import time
from flask import Flask, request, jsonify, Blueprint
from flask_restful import reqparse, abort, Api, Resource, marshal_with
import uuid
from my_model import *
import base64

app = Flask(__name__)
api_bp = Blueprint('api', __name__)
api = Api(api_bp)
# api = Api(app)

parser = reqparse.RequestParser(bundle_errors = True)
parser.add_argument('data', type=str, required=True, help='Content cannot be left blank')

def convertImgToBase64(img_path):
	with open(img_path, 'rb') as f:
		return base64.encodebytes(f.read()).decode('ascii')


class HandwrittingOCR(Resource):
    def post(self):
        data = {}
        try:
            args = parser.parse_args()
            text =  args.data
            timesteps = (len(text) * 20)
            timesteps = timesteps - (timesteps%8) + 8 
            res = my_utils.run_batch_inference(model, beta_set, text, style_vector, 
                            tokenizer=tokenizer, time_steps=timesteps, diffusion_mode=args_diffmode, 
                            show_samples=False, path='test_api')
            data = {'image': convertImgToBase64('test_api.png')}
        except Exception as e:
            data = {'error': 200}
        return jsonify(data)


api.add_resource(HandwrittingOCR,'/')
app.register_blueprint(api_bp)

if __name__=='__main__':
    app.run(host = '0.0.0.0', debug=True, port='2112', use_reloader=False)