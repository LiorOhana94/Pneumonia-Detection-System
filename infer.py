import os
from flask import Flask, jsonify, request
from argparse import ArgumentParser
import uuid
from predict import model
from predict import predict as pred
import torch


parser = ArgumentParser()
parser.add_argument("-m", "--model", dest="model",
                    help="location of the pickle file")

filename = parser.parse_args().model

app = Flask(__name__)
print("hi")
@app.route('/')
def index():
    return "ELI - OHANA =]"

@app.route('/predict/<int:x>', methods=['GET'])
def predict(x):
	
	return ("Fuck you % times (" + filename + ")" % (x))

@app.route('/uploader', methods =  ['POST'])
def upload_file():
   if request.method == 'POST':

      f = request.files['file']
      print(f.filename)
      extension = os.path.splitext(f.filename)[1]
      filename = uuid.uuid4()
      path = 'temp-images/%s%s' % (filename, extension)
      f.save(path)
      res = pred(model, path)[0]
      print(res)
      return "%d" % res

if __name__ == '__main__':
      app.run(host='0.0.0.0', port=8080, debug=True)