import os
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

from flask import Flask, jsonify, request, abort
from argparse import ArgumentParser
import uuid
from model import model
from predict import predictResnet as pred
import torch
import requests
from io import BytesIO

labels = ["healthy", "pneumonia"]

parser = ArgumentParser()
parser.add_argument("-m", "--model", dest="model",
                    help="location of the pickle file")

filename = parser.parse_args().model


app = Flask(__name__)

@app.route('/')
def index():
    return "ELI - OHANA =]"

@app.route('/predict', methods =  ['POST'])
def upload_file():
   if request.method == 'POST':
      response = requests.get(request.json['scan'])
      scan_guid = request.json['scanGuid']
      res, prob = pred(model, BytesIO(response.content), scan_guid, generate_map=True)
      print(res)
      results = { 'result_index': res.tolist(), 'result_prob': prob.tolist(), 'result_text': labels[res], 'heatmap_guid': scan_guid}
      return jsonify(results)

@app.route('/send-activation-map', methods =  ['POST'])
def send_file():
      scan_giud = request.json['scanGuid']
      destination = request.json['destination']
      print(scan_giud)
      if scan_giud is None: 
            abort(405)
      map_path = f'./class-activation-maps/{scan_giud}.cam.png'
      
      if not(os.path.exists(map_path)):
            abort(405)

      files = {'map':  (f'{scan_giud}.map.png', open(map_path, 'rb'), 'image/png')}
      requests.post(destination, files=files)
      return "ok"


if __name__ == '__main__':
      app.run(host='0.0.0.0', port=8080, debug=True)

      """      
      f = request.files['file']
      extension = os.path.splitext(f.filename)[1]
      filename = uuid.uuid4()
      path = 'temp-images/%s%s' % (filename, extension)
      f.save(path)"""