from flask import Flask
from deepforest import deepforest

UPLOAD_FOLDER = './static/uploads/'

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

#model = deepforest.deepforest()
#model.use_release()
