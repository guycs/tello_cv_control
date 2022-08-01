import logging

from flask import jsonify
from flask import render_template
from flask import request
from flask import Response

from droneapp.models.drone_manager import DroneManager

import config


logger = logging.getLogger(__name__)
app = config.app


def get_drone():
    return DroneManager()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/controller/')
def controller():
    return render_template('controller.html')


@app.route('/api/command/', methods=['POST'])
def command():
    cmd = request.form.get('command')
    logger.info({'action': 'command', 'cmd': cmd})
    result = None
    drone = get_drone()
    if cmd == 'takeOff':
        result = drone.takeoff()
    if cmd == 'land':
        result = drone.land()
    if cmd == 'speed':
        speed = request.form.get('speed')
        logger.info({'action': 'command', 'cmd': cmd, 'speed': speed})
        if speed:
            result = drone.set_speed(int(speed))

    if cmd == 'up':
        result = drone.up()
    if cmd == 'down':
        result = drone.down()
    if cmd == 'forward':
        result = drone.forward()
    if cmd == 'back':
        result = drone.back()
    if cmd == 'clockwise':
        result = drone.clockwise()
    if cmd == 'counterClockwise':
        result = drone.counter_clockwise()
    if cmd == 'left':
        result = drone.left()
    if cmd == 'right':
        result = drone.right()
    if cmd == 'flipFront':
        result = drone.flip_front()
    if cmd == 'flipBack':
        result = drone.flip_back()
    if cmd == 'flipLeft':
        result = drone.flip_left()
    if cmd == 'flipRight':
        result = drone.flip_right()
    if cmd == 'patrol':
        result = drone.patrol()
    if cmd == 'stopPatrol':
        result = drone.stop_patrol()
    if cmd == 'faceDetectAndTrack':
        result = drone.enable_face_detect()
    if cmd == 'stopFaceDetectAndTrack':
        result = drone.disable_face_detect()
    if cmd == 'snapshot':
        if drone.snapshot():
            return jsonify(status='success'), 200
        else:
            return jsonify(status='fail'), 400

    return jsonify(status = result), 200

def video_generator():
    drone = get_drone()
    for jpeg in drone.video_jpeg_generator():
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               jpeg +
               b'\r\n\r\n')


@app.route('/video/streaming')
def video_feed():
    return Response(video_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run():
    app.run(host=config.WEB_ADDRESS, port=config.WEB_PORT, threaded=True)