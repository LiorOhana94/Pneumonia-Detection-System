import os
from flask import Flask, jsonify, request, abort
from argparse import ArgumentParser
import uuid
from model import model
from predict import predict as pred
import torch
import requests

labels = ["healthy", "pneumonia"]

parser = ArgumentParser()
parser.add_argument("-m", "--model", dest="model",
                    help="location of the pickle file")

filename = parser.parse_args().model

app = Flask(__name__)

@app.route('/')
def index():
    return "ELI - OHANA =]"

@app.route('/diagnose', methods =  ['POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      extension = os.path.splitext(f.filename)[1]
      filename = uuid.uuid4()
      path = 'temp-images/%s%s' % (filename, extension)
      f.save(path)
      res, heatmap_file_name  = pred(model, path, generate_map=True)
      print(res)
      results = { 'result_index': res.tolist(), 'result_text': labels[res], 'heatmap_file_name': heatmap_file_name}
      return jsonify(results)

@app.route('/send-activation-map', methods =  ['GET'])
def send_file():
      scan_giud = request.args.get('scan_guid')
      print(scan_giud)
      if scan_giud is None: 
            abort(405)
      map_path = f'./class-activation-maps/{scan_giud}.png'
      
      if not(os.path.exists(map_path)):
            abort(405)

      files = {'file':  (f'{scan_giud}.png', open(map_path, 'rb'), 'image/png')}
      requests.post("http://127.0.0.1:3000/upload-image/", files=files)
      return "ok"


if __name__ == '__main__':
      app.run(host='0.0.0.0', port=8080, debug=True)