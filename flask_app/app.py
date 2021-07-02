import numpy as np
import pandas as pd
import joblib
import sys
import json
import io
from flask import Flask, escape, jsonify, request, make_response, current_app, send_file, send_from_directory
from datetime import timedelta, datetime
from functools import update_wrapper
import os
import logging
import pathlib
import pickle
from joblib import dump, load
from random import sample
import sklearn
from sklearn import datasets, linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from PIL import Image
from base64 import decodebytes, encodebytes, decodestring, encodestring
import cv2
#from cv2 import cv2
from keras.models import load_model
import tensorflow as tf
from GazeML_keras import diagnose
#from EyeDiagnosisLib import GazeML_keras

server = Flask(__name__)


def crossdomain(origin=None, methods=None, headers=None, max_age=21600,
                attach_to_all=True, automatic_options=True):
    """Decorator function that allows crossdomain requests.
      Courtesy of
      https://blog.skyred.fi/articles/better-crossdomain-snippet-for-flask.html
    """
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    # use str instead of basestring if using Python 3.x
    if headers is not None and not isinstance(headers, str):
        headers = ', '.join(x.upper() for x in headers)
    # use str instead of basestring if using Python 3.x
    if not isinstance(origin, str):
        origin = ', '.join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        """ Determines which methods are allowed
        """
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        """The decorator function
        """
        def wrapped_function(*args, **kwargs):
            """Caries out the actual cross domain code
            """
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp

            h = resp.headers
            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            h['Access-Control-Allow-Credentials'] = 'true'
            h['Access-Control-Allow-Headers'] = \
                "Origin, X-Requested-With, Content-Type, Accept, Authorization"
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)
    return decorator


def run_request():
    json_data = request.json
    a_value = json_data["a_key"]
    return "JSON value sent: " + a_value


def normalize_input(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    # x should be RGB with range [0, 255]
    return ((x / 255) - mean) / std


@server.route('/', methods=['GET', 'POST'])
def main_get():
    if request.method == 'GET':
        return 'The InnoSpark models are up and running. GET is not degined at this moment. Send a POST request LVEA'
    else:
    	#return 'The InnoSpark models are up and running. POST request'
        return run_request()


@server.route('/predictrfc', methods=['GET', 'POST'])
def predictrfc():
    """API Call
    """
    try:
        #app.logger.warning('testing warning log')
        print('Entering predictrfc method', file=sys.stderr)

        #Reading request json
        requestDict = request.get_json()
        requestJson = json.dumps(requestDict)
        print(type(requestJson), file=sys.stderr)
        print(requestJson, file=sys.stderr)

        dfJson = pd.read_json(requestJson, orient='split')
        print(dfJson)

        #load model
        rfc = joblib.load('./models/rfc.pkl')
        predictrfc = rfc.predict(dfJson)

        #predictrfc['X'] = predictrfc.index
        #predictrfc = predictrfc.rename(columns={-1: "Y1", -2: "Y2", "X": "X"})

        print(predictrfc.shape)
        print(predictrfc.columns)

        predictionDic = predictrfc.to_dict('records')

        responses = jsonify(predictions=predictionDic)
        responses.status_code = 200

    except Exception as e:
        responses = jsonify(predictions={
                            'error': 'some error occured, please try again later!'}, Exception=e)
        responses.status_code = 404
        #print ('error', e, file=sys.stderr)
    return (responses)


@server.route('/predictrfc2', methods=['GET', 'POST'])
def predictrfc2():
    """API Call
    """
    #app.logger.warning('testing warning log')
    print('Entering predictrfc method', file=sys.stderr)

    #Reading request json
    requestDict = request.get_json()
    requestJson = json.dumps(requestDict)
    print(type(requestJson), file=sys.stderr)
    print(requestJson, file=sys.stderr)

    dfJson = pd.read_json(requestJson, orient='records')
    print(dfJson)

    #
    print(pathlib.Path(__file__).parent.absolute())
    print(pathlib.Path().absolute())
    #load model
    curr_dir_path = pathlib.Path(__file__).parent.absolute()
    model_path = '/models/regr.pkl'
    # Convert path to Windows format
    model_path_win = pathlib.PureWindowsPath(model_path)
    a = str(curr_dir_path) + str(model_path_win)
    print(sklearn.__version__)
    regr = joblib.load(a)
    predictrfc = regr.predict(dfJson)

    #predictrfc['X'] = predictrfc.index
    #predictrfc = predictrfc.rename(columns={-1: "Y1", -2: "Y2", "X": "X"})

    #print(predictrfc.shape)
    #print(predictrfc.columns)

    print(type(predictrfc))
    print(predictrfc)
    #predictionDic = dict(np.ndenumerate(predictrfc))

    #predictionDic = predictrfc.to_dict('records')

    #responses = jsonify(predictions=predictionDic)
    responses = jsonify(predictions=predictrfc.tolist())
    responses.status_code = 200

    return (responses)


@server.route('/predicteyedisease', methods=['POST'])
def predicteyedisease():
    """API Call
    """
    #app.logger.warning('testing warning log')
    print('Entering predicteyedisease method', file=sys.stderr)
    #img = Image.open(request.files['file'])
    request_file = request.files['file']
    request_image = Image.open(request_file)
    numpydata = np.asarray(request_image)

    # <class 'numpy.ndarray'>
    #print(type(numpydata))

    #  shape
    #print(numpydata.shape)

    #resized_img = cv2.resize(numpydata[:, :, ::-1], (64, 64))
    #reshaped_img = resized_img.reshape(1, 64, 64, -1)

    resized_img = cv2.resize(numpydata[:, :, ::-1], (100, 100))
    reshaped_img = resized_img.reshape(1, 100, 100, -1)

    #load model
    curr_dir_path = pathlib.Path(__file__).parent.absolute()
    #model_path = '/models/sigmoid_final_cataract-90.h5'
    model_path = '/models/accuracy-92size100x100_Threshold_0.56.h5'
    # Convert path to Windows format
    #model_path_win = pathlib.PureWindowsPath(model_path)

    model_path_full = str(curr_dir_path) + str(model_path)

    #import keras model
    eye_model = load_model(model_path_full)

    cataract_pred = eye_model.predict(reshaped_img)
    cataract_pred_per = float(cataract_pred) * 100

    #responses = jsonify(predictions=predictrfc.tolist())
    responses = jsonify(predictions=cataract_pred_per)
    responses.status_code = 200

    return (responses)


@server.route('/parseimage', methods=['POST'])
def parseimage():
    """API Call
    """
    #app.logger.warning('testing warning log')
    print('Entering parseimage method', file=sys.stderr)
    #img = Image.open(request.files['file'])
    request_file = request.files['file']
    request_image = Image.open(request_file)
    numpydata = np.asarray(request_image)

    # <class 'numpy.ndarray'>
    #print(type(numpydata))

    #  shape
    #print(numpydata.shape)

    #resized_img= cv2.resize(numpydata[:,:,::-1], (64, 64))
    #reshaped_img = resized_img.reshape(1 ,64 , 64 , -1)

    im = numpydata[..., ::-1]
    #print("Fists:")
    #print(im.shape)
    face = im
    orig_h, orig_w = face.shape[:2]
    inp = cv2.resize(face, (512, 512))
    inp = normalize_input(inp)
    inp = inp[None, ...]
    #print("second:")
    #print(inp.shape)

    #load model
    curr_dir_path = pathlib.Path(__file__).parent.absolute()
    model_path = '/models/parser_net.h5'
    # Convert path to Windows format
    #model_path_win = pathlib.PureWindowsPath(model_path)

    model_path_full = str(curr_dir_path) + str(model_path)

    #import keras model
    parse_img_model = load_model(model_path_full, custom_objects={'tf': tf})

    parse_img_pred = parse_img_model.predict([inp])[0]
    #print("parse_img_pred:")
    #print(parse_img_pred.shape)

    parsing_map = parse_img_pred.argmax(axis=-1)

    #print("third:")
    #print(parsing_map.shape)
    parsing_map = cv2.resize(parsing_map.astype(
        np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    #print("fourth:")
    #print(parsing_map.shape)

    #Parsed images directory
    parsed_img_dir = '/parsed_images/'
    parsed_img_dir_full = str(curr_dir_path) + str(parsed_img_dir)

    #tmp_var =  parsing_map.tolist()
    #np.savetxt('vb_array.out', parsing_map, delimiter=',')

    print("parsing_map:")
    print(parsing_map.shape)

    #img_parse = Image.fromarray(parsing_map, 'RGB')
    img_parse = Image.fromarray(parsing_map)
    img_parse_name_tmp = datetime.now().strftime("%Y%m%d%H%M%S") + '.jpg'
    tmp_img_name = parsed_img_dir_full + img_parse_name_tmp
    img_parse.save(tmp_img_name,  quality=100)

    #TODO: Delete Block
    #img_parse_ori = Image.fromarray(numpydata)
    #img_parse_name_tmp_ori = datetime.now().strftime("%Y%m%d%H%M%S") + '_ori'+ '.jpg'
    #tmp_img_name_ori = parsed_img_dir_full + img_parse_name_tmp_ori
    #img_parse_ori.save(tmp_img_name_ori,  quality=100)

    #ImgBin =  Image.open(tmp_img_name).tobytes()

    #return send_file(
    #    io.BytesIO(ImgBin),
    #    mimetype='image/png',
    #    as_attachment=True,
    #    attachment_filename=tmp_img_name)

    return send_from_directory(parsed_img_dir_full, img_parse_name_tmp, as_attachment=True)

def get_response_image(byte_arr):
    pil_img = Image.fromarray(np.uint8(byte_arr)).convert('RGB')
    byte_arr_io = io.BytesIO()
    pil_img.save(byte_arr_io, format='PNG') # convert the PIL image to byte array
    #encoded_img = encodestring(byte_arr_io.getvalue()) # encode as base64
    encoded_img = encodebytes(byte_arr_io.getvalue()).decode('ascii') # encode as base64
    #encoded_img = encodebytes(np.array(pil_img)).decode('ascii') # encode as base64
    return encoded_img

#This method is just for testing image decoding
def write_reponse_image(image_64_encode):
    #Parsed images directory
    curr_dir_path = pathlib.Path(__file__).parent.absolute()
    parsed_img_dir = '/parsed_images/'  
    parsed_img_dir_full = str(curr_dir_path) + str(parsed_img_dir)

    #Decoding image_64_encode
    #image_64_decode = decodestring(image_64_encode)
    image_64_decode = decodebytes(image_64_encode.encode("ascii"))
    image_result_name_tmp = datetime.now().strftime("%Y%m%d%H%M%S%f") + '.jpg'
    tmp_img_name = parsed_img_dir_full + image_result_name_tmp
    image_result = open(tmp_img_name, 'wb') # create a writable image and write the decoding result
    image_result.write(image_64_decode)

@server.route('/eyesDiagnosis', methods=['POST'])
def eyesDiagnosis():
    try:
        print('Entering eyediagnosis method', file=sys.stderr)

        data = []
        response_msg = ""
        status_code = 200

        request_file = request.files['file']
        request_image = Image.open(request_file)
        
        numpydata = np.array(request_image)
        if numpydata.ndim >  2 and numpydata.shape[2] > 3:
            numpydata = numpydata[...,:3]
        else:
            numpydata = numpydata[...,::-1]

        #numpydata = np.asarray(request_image)[...,::-1]

        detect_model_path = 'mtcnn_weights'
        diagnosis_model_path='accuracy-92size100x100_Threshold_0.56.h5'
        left_eye_im , left_eye_im_desc , left_eye_im_diagnosis , right_eye_im , right_eye_im_desc , right_eye_im_diagnosis = diagnose.Diagnose().Diagnose_patient(numpydata,detect_model_path,diagnosis_model_path)
        
        #print(f"Response msg:{response_msg}")
        if isinstance(left_eye_im, str) or isinstance(right_eye_im, str):
            response_msg = left_eye_im
        else:
            #TODO: Comment these two lines, they're created just to test the way the front end can consume this endpoint
            #write_reponse_image(get_response_image(left_eye_im))
            #write_reponse_image(get_response_image(right_eye_im))
            
            response_msg = "success"

            data.append({
            "left_eye_im": get_response_image(left_eye_im),
                "left_eye_im_desc": left_eye_im_desc,
                "left_eye_im_diagnosis": str(left_eye_im_diagnosis[0]),
                "right_eye_im": get_response_image(right_eye_im),
                "right_eye_im_desc":right_eye_im_desc,
                "right_eye_im_diagnosis":str(right_eye_im_diagnosis[0])
            })
    except Exception as e:
        status_code = 400
        response_msg = e.message
        #    responses = jsonify(
        #        message="failure " + e.message,
        #        category="prediction",
        #        data=[],
        #        status=400
        #    )
        #    responses.status_code = 400
    finally:
        responses = jsonify(
            message=response_msg,
            category="prediction",
            data=data,
            status=status_code
        )
        responses.status_code = status_code

    return (responses)



def main():
    # use 0.0.0.0 to use it in container
    server.run(host='0.0.0.0')

#    #app.run(debug=True)
#    print (f"app_print_main:{__name__}", file=sys.stderr)
#    if __name__ == '__main__':
#        app.run()
    
 #       logging.basicConfig(filename='myapp.log', level=logging.DEBUG)
 #       logging.info('Started')
 #       #mylib.do_something()
 #       logging.info('Finished')


if __name__ == '__main__':
    main()