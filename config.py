import os
import numpy as np

from flask import Flask


WEB_ADDRESS = '0.0.0.0'
WEB_PORT = 6500
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TEMPLATES = os.path.join(PROJECT_ROOT, 'droneapp/templates')
STATIC_FOLDER = os.path.join(PROJECT_ROOT, 'droneapp/static')
DEBUG = False
LOG_FILE = 'pytello.log'

LOWER_BLUE = np.array([90,20,50])
UPPER_BLUE = np.array([110,255,255])

CAMERA_SIZE_X = 640
CAMERA_SIZE_Y = 480


#CAMERA_SIZE_X = 1280    #960
#CAMERA_SIZE_Y = 720


FRAME_SHRINK_RATIO = 1

FRAME_SIZE_X = CAMERA_SIZE_X // FRAME_SHRINK_RATIO
FRAME_SIZE_Y = CAMERA_SIZE_Y // FRAME_SHRINK_RATIO

FRAME_SIZE_AREA = FRAME_SIZE_X * FRAME_SIZE_Y

app = Flask(__name__,
            template_folder=TEMPLATES,
            static_folder=STATIC_FOLDER)
if DEBUG:
    app.debug = DEBUG
