#--Libraries
from flask import Flask, request, jsonify, request
from flask_cors import CORS
import os
import json

#--Routing Files--
from routes.app import app_routes

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
os.chdir(dir_path)
with open('./static/paths.json') as p:
    paths = json.load(p)

#--Flask init--
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = paths['upload_dir']
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
cors = CORS(app)

#--Routes--
@app.route('/app/<url>', methods=['GET', 'POST'])
def apps(url):
    return app_routes(url, app, request)

if __name__ == '__main__':
    app.run(host='localhost', port=1113)


