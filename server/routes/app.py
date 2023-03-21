from os import listdir
from os.path import join
from werkzeug.utils import secure_filename
from flask import jsonify
from modules.detect_image import detect_image
import json

with open('./static/paths.json') as p:
    paths = json.load(p)

#--Methods--
def listmodels():
    models_list = [m for m in listdir(paths['models_dir'])]
    response = jsonify(models_list)
    return response

def save_image(file):
    ext = file.filename.split('.')[-1]
    fname = f'img_upload.{ext}'
    save_dir = join(_app.config['UPLOAD_FOLDER'], secure_filename(fname))
    file.save(save_dir)
    return save_dir

def identify():
    global _req
    global _app
    img_dir = save_image(_req.files['file'])
    param = json.loads(_req.form['param'])

    response = jsonify(detect_image(param['model'], float(param['consistency']), float(param['uniqueness']), img_dir))

    return response

#--Routes--
routes = {
    'models': listmodels,
    'identify': identify
}

#--app & request object--
_app = None
_req = None

def app_routes(url, app, req):
    global routes
    global _req
    global _app

    _req = req
    _app = app

    return routes[url]()