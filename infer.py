from flask import Flask, jsonify
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-m", "--model", dest="model",
                    help="location of the pickle file")

filename = parser.parse_args().model

app = Flask(__name__)

@app.route('/')
def index():
    return "ELI - OHANA =]"

@app.route('/predict/<int:x>', methods=['GET'])
def predict(x):
	
	return "Fuck you " + x + " times (" + filename + ")"

if __name__ == '__main__':
      app.run(host='0.0.0.0', port=8080)