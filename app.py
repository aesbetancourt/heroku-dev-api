# Importing the dependencies
from flask import Flask, request
from tensorflow.keras.models import load_model
from json import dumps
import numpy as np

error_msg = ""
error = 0
# Check if all values are good
def check_lists(arr):
    global error_msg
    global error
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


# Return a prediction from the model given args
def get_prediction(model, input_json, names):
    dev_quality = input_json[names[0]]
    dev_on_time = input_json[names[1]]
    team_chemistry = input_json[names[2]]
    dev_exp = input_json[names[3]]
    pro_exp = input_json[names[4]]
    developer = np.array([[dev_quality, dev_on_time, team_chemistry, dev_exp, pro_exp]])
    success = "{:.0f}".format(float(str(model.predict(developer)[0][0])))
    return success


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
    err = 0
    err_msg = ""

    args = [
        request.args.get('model'),
        request.args.get('arg1'),  # M1:
        request.args.get('arg2'),  #
        request.args.get('arg3'),  #
        request.args.get('arg4'),  #
        request.args.get('arg5'),  #
        request.args.get('arg6')   #
    ]

    # args = [request.args.get('qy'), request.args.get('tms'), request.args.get('tch'),
    #         request.args.get('sk'), request.args.get('rqs'), request.args.get('num')]
    #
    # args = [request.args.get('texp'), request.args.get('mexp'), request.args.get('len'),
    #         request.args.get('ent'), request.args.get('langs')]
    # Handling exceptions

    # model1_info = list()
    # model2_info = list()

    if args[0] == "1":
        model = 'dev_model'
        model1_info = [args[1], args[2], args[3], args[4], args[5], args[6]]
        err, err_msg = check_lists(model1_info)
        headers = ['code_quality', 'dev_on_time', 'team_chemistry', 'dev_exp', 'pro_exp', 'num']
        # print(err, err_msg, model1_info)
    elif args[0] == "2":
        model = 'effort_model'
        model2_info = [args[1], args[2], args[3], args[4], args[5]]
        err, err_msg = check_lists(model2_info)
        headers = ['team_exp', 'manager_exp', 'lenght', 'entities', 'language']
    else:
        err = 1
        err_msg = "Model selection error"

    content['ERROR_STATUS'] = err
    content['DNN_MODEL'] = args[0]
    content['SELECTED_MODEL'] = model
    if not (content['ERROR_STATUS']):
        for i in range(len(headers)):
            content[headers[i]] = float(args[i+1])

            # content['code_quality'] = float(args[0])
            # content['dev_on_time'] = float(args[1])
            # content['team_chemistry'] = float(args[2])
            # content['dev_exp'] = float(args[3])
            # content['pro_exp'] = float(args[4])
            # content['num'] = int(args[5])
        # If there isn't any error then ask to the model
        if content['DNN_MODEL'] == '1':
            result = get_prediction(model=dev_model, input_json=content, names=headers)
        else:
            result = get_prediction(model=effort_model, input_json=content, names=headers)
        content['accomplishment'] = result
    else:
        content['ERROR_MSG'] = err_msg
    json_data = dumps(content)
    return json_data


if __name__ == '__main__':
    app.run(debug=True)
