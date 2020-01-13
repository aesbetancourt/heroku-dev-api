# Importing the dependencies
from flask import Flask, request
from tensorflow.keras.models import load_model
from json import dumps
import numpy as np


# Return a prediction from the model given args
def get_prediction(model, input_json):
    dev_quality = input_json['code_quality']
    dev_on_time = input_json['dev_on_time']
    team_chemistry = input_json['team_chemistry']
    dev_exp = input_json['dev_exp']
    pro_exp = input_json['pro_exp']
    developer = np.array([[dev_quality, dev_on_time, team_chemistry, dev_exp, pro_exp]])
    success = "{:.0f}".format(float(str(model.predict(developer)[0][0])))

    return success


# Creating a Flask instance app
app = Flask(__name__)

# Loading the model
dev_model = load_model('model.h5')


# API
# This is a sample link to get results in JSON format from the API
# http://host:port/?quality=23&times=68&chemistry=53&skills=40&reqs=80
@app.route('/')
# Return a json containing the given params and the prediction result
def prediction():
    # Defining content dictionary
    content = dict()
    content['code_quality'] = float(request.args.get('quality'))
    content['dev_on_time'] = float(request.args.get('times'))
    content['team_chemistry'] = float(request.args.get('chemistry'))
    content['dev_exp'] = float(request.args.get('skills'))
    content['pro_exp'] = float(request.args.get('reqs'))
    # Ask to the model
    result = get_prediction(model=dev_model, input_json=content)
    content['accomplishment'] = result
    json_data = dumps(content)
    return json_data


if __name__ == '__main__':
    app.run(debug=True)
