# Importing the dependencies
from flask import Flask, request
from tensorflow.keras.models import load_model
from json import dumps
import numpy as np


# Return a prediction from the model given args
def get_prediction(model, input_json, names):
    arg1 = input_json[names[0]]
    arg2 = input_json[names[1]]
    arg3 = input_json[names[2]]
    arg4 = input_json[names[3]]
    arg5 = input_json[names[4]]
    input_features = np.array([[arg1, arg2, arg3, arg4, arg5]])
    prediction_result = "{:.0f}".format(float(str(model.predict(input_features)[0][0])))
    return prediction_result


# Check if all values are good
def check_lists(arr):
    error = 0
    error_msg = ""
    for arg in arr:
        # NoneType Object
        if arg is None:
            error = 1
            try:
                raise TypeError("Cannot support NoneType object. Missing arguments.")
            except TypeError as e:
                error_msg = str(e)
            break
        # Not a number
        try:
            arg = float(arg)
        except ValueError as e:
            error = 1
            error_msg = str(e)
            break
    return error, error_msg


# Creating a Flask instance app
app = Flask(__name__)

# Loading the models
dev_model = load_model('models/dev_predictor.h5')
effort_model = load_model('models/effort_predictor.h5')


# API
# This is a sample link to get results in JSON format from the API

# Model 1: Dev predictor
# /?model=1&arg1=23&arg2=68&arg3=53&arg4=40&arg5=80&arg6=1

# Model 2: Effort predictor
# /?model=2arg1=23&arg2=68&arg3=53&arg4=40&arg5=80
@app.route('/')
# Return a json containing the given params and the prediction result
def prediction():
    # Defining content dictionary
    content = dict()

    # Get the params sent in the url
    args = [
        request.args.get('model'),  # Model selection parameter (1, 2)
        request.args.get('arg1'),  # M1: code quality         # M2: Lang exp %
        request.args.get('arg2'),  # M1: dev on time          # M2: manager experience %
        request.args.get('arg3'),  # M1: team chemistry       # M2: lenght Years
        request.args.get('arg4'),  # M1: dev experience       # M2: entities Num
        request.args.get('arg5'),  # M1: project experience   # M2: Team exp %
        request.args.get('arg6')   # M1: developer number (App Helper, not required in model 2)
    ]

    if args[0] == "1":
        model_name = 'dev_model'
        model1_data = [args[1], args[2], args[3], args[4], args[5], args[6]]
        err, err_msg = check_lists(model1_data)
        headers = ['code_quality', 'dev_on_time', 'team_chemistry', 'dev_exp', 'pro_exp', 'num']
    elif args[0] == "2":
        model_name = 'effort_model'
        model2_data = [args[1], args[2], args[3], args[4], args[5]]
        err, err_msg = check_lists(model2_data)
        headers = ['team_exp', 'manager_exp', 'lenght', 'entities', 'language']
    else:
        err = 1
        err_msg = "Model selection error"
        model_name = "No selected Model"
        headers = []

    content['ERROR_STATUS'] = err
    content['DNN_MODEL'] = args[0]
    content['SELECTED_MODEL'] = model_name
    if not (content['ERROR_STATUS']):
        for i in range(len(headers)):
            content[headers[i]] = float(args[i + 1])
        # If there isn't any error then ask to the selected model
        if content['DNN_MODEL'] == '1':
            result = get_prediction(model=dev_model, input_json=content, names=headers)
        else:
            result = get_prediction(model=effort_model, input_json=content, names=headers)
        content['result'] = result
    else:
        content['ERROR_MSG'] = err_msg
    json_data = dumps(content)
    return json_data


if __name__ == '__main__':
    app.run(debug=True)
