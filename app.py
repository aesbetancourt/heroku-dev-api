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
# http://host:port/?qy=23&tms=68&tch=53&sk=40&rqs=80
@app.route('/')
# Return a json containing the given params and the prediction result
def prediction():
    # Defining content dictionary
    content = dict()
    err_msg = ""
    err = 0
    args = [request.args.get('qy'), request.args.get('tms'), request.args.get('tch'),
            request.args.get('sk'), request.args.get('rqs'), request.args.get('num')]
    # Handling exceptions
    for arg in args:
        # NoneType Object
        if arg is None:
            err = 1
            try:
                raise TypeError("Cannot support NoneType object. Missing arguments.")
            except TypeError as e:
                err_msg = str(e)
            break
        # Not a number
        try:
            arg = float(arg)
        except ValueError as e:
            err = 1
            err_msg = str(e)
            break

    content['ERROR_STATUS'] = err
    if not(content['ERROR_STATUS']):
        content['code_quality'] = float(args[0])
        content['dev_on_time'] = float(args[1])
        content['team_chemistry'] = float(args[2])
        content['dev_exp'] = float(args[3])
        content['pro_exp'] = float(args[4])
        content['num'] = int(args[5])
        # If there isn't any error then ask to the model
        result = get_prediction(model=dev_model, input_json=content)
        content['accomplishment'] = result
    else:
        content['ERROR_MSG'] = err_msg
    json_data = dumps(content)
    return json_data


if __name__ == '__main__':
    app.run(debug=True)
