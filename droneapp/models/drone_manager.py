import logging
import contextlib
import os
import socket
import subprocess
import threading
import time
import config

import cv2 as cv
import numpy as np
import win32com.client as wincl

from droneapp.models.base import Singleton
from droneapp.models.findSquare import Tracker
import pythoncom
import serial

logger = logging.getLogger(__name__)

DEFAULT_DISTANCE = 0.30
DEFAULT_SPEED = 10
DEFAULT_DEGREE = 10

FRAME_SHRINK_FACTOR = 3
FRAME_X = config.FRAME_SIZE_X
FRAME_Y = config.FRAME_SIZE_Y
FRAME_AREA = FRAME_X * FRAME_Y

FRAME_SIZE = FRAME_AREA * 3
FRAME_CENTER_X = FRAME_X / 2
FRAME_CENTER_Y = FRAME_Y / 2

CMD_FFMPEG = (f'ffmpeg -hwaccel auto -hwaccel_device opencl -i pipe:0 '
              f'-pix_fmt bgr24 -s {FRAME_X}x{FRAME_Y} -f rawvideo pipe:1')

FACE_DETECT_XML_FILE = './droneapp/models/haarcascade_frontalface_default.xml'

SNAPSHOT_IMAGE_FOLDER = './droneapp/static/img/snapshots/'

class ErrorNoFaceDetectXMLFile(Exception):
    """Error no face detect xml file"""


class ErrorNoImageDir(Exception):
    """Error no image dir"""

class MissionState:
    def __init__(self):
        self.message = 'start'
        self.standby = False
        self.vertical_take_off = False
        self.docking_take_off = False
        self.fly_to_destination = False
        self.dock = False
        self.charge = False
        self.vertical_landing = False
        self.follow_me = False
        self.eject = False
        self.failed = False
        self.test = False
        self.test_servos = True

class DockingState:
    def __init__(self,
            message = 'start',
            start = False,
            approach = False,
            search_dock = False,
            get_to_run_base = False,
            docking_run = False,
            blind_stretch = False,
            flip = False,
            end = False,
            failed = False
                 ):

        self.message = message
        self.start = start
        self.approach = approach
        self.search_dock = search_dock
        self.get_to_run_base = get_to_run_base
        self.docking_run = docking_run
        self.blind_stretch = blind_stretch
        self.flip = flip
        self.end = end
        self.failed = failed


class DroneStatus:
    def __init__(self):
        battery = None

class DroneManager(metaclass=Singleton):
    def __init__(self, host_ip='192.168.10.2', host_port=8889,
                 drone_ip='192.168.10.1', drone_port=8889,
                 is_imperial=False, speed=DEFAULT_SPEED):
        self.host_ip = host_ip
        self.host_port = host_port
        self.drone_ip = drone_ip
        self.drone_port = drone_port
        self.drone_address = (drone_ip, drone_port)
        self.reconnecting = False
        self.is_imperial = is_imperial
        self.speed = speed
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.bindSuccess = True
        self.frame_counter = 0
        self.lastCmd = False
        try:
            self.socket.bind((self.host_ip, self.host_port))
        except :
            logger.warning('couldnt bind to drone' )
            self.bindSuccess = False

        self.response = None
        self.stop_event = threading.Event()
        self._response_thread = threading.Thread(target=self.receive_response,
                                           args=(self.stop_event, ))
        self._response_thread.start()

        self.patrol_event = None
        self.is_patrol = False
        self.is_auto_land = False
        self.auto_land_stage = 'start'
        self._patrol_semaphore = threading.Semaphore(1)
        self._thread_patrol = None

        self.proc = subprocess.Popen(CMD_FFMPEG.split(' '),
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE)
        self.proc_stdin = self.proc.stdin
        self.proc_stdout = self.proc.stdout

        self.video_port = 11111

        self._receive_drone_video_thread = threading.Thread(
            target=self.receive_drone_video,
            args=(self.stop_event, self.proc_stdin,
                  self.host_ip, self.video_port,))
        # comeback
        self._receive_drone_video_thread.start()

        if not os.path.exists(FACE_DETECT_XML_FILE):
            raise ErrorNoFaceDetectXMLFile(f'No {FACE_DETECT_XML_FILE}')
        self.face_cascade = cv.CascadeClassifier(FACE_DETECT_XML_FILE)
        self._is_enable_face_detect = False

        if not os.path.exists(SNAPSHOT_IMAGE_FOLDER):
            raise ErrorNoImageDir(f'{SNAPSHOT_IMAGE_FOLDER} does not exists')
        self.is_snapshot = False

        self._command_semaphore = threading.Semaphore(1)
        self._command_thread = None
        if (self.bindSuccess):
            self.send_command('command', expectResponse=True)
            self.send_command('streamon',  expectResponse=True)
            self.set_speed(self.speed)

        self.lastFrame = None
        self.frameUpdated = False
        self.frameUpdatedForMissinManager = False
        self.tracker = Tracker(debug=False)
        self.track_target = False
        self.mission_state = MissionState()
        self._video_processing_thread = threading.Thread(
            target=self.video_processing,
            args=(self.stop_event,))
        self._video_processing_thread.start()

        self.drone_status = DroneStatus()
        self._get_drone_status_thread = threading.Thread(target=self.get_drone_status,
                                                   args=(self.stop_event,))
        # comeback
        self._get_drone_status_thread.start()


        self._manage_mission_thread = threading.Thread(target=self.manage_mission,
                                                         args=(self.stop_event,))
        # comeback
        self._manage_mission_thread.start()
        self.ser = serial.Serial('COM8', 9600)  # ('/dev/ttyUSB0')  # open serial port

    def control_servos(self, position):
        logger.info(f' connecting to port {self.ser.name}')  # check which port was really used
        self.ser.write(position)  # write a string
        time.sleep(1)
        # response = ser.read(100)
        #logger.info("servos response: " + response)
        #ser.close()  # close port

    def get_drone_status(self, stop_event):
        while not stop_event.is_set():
            try:
                self.drone_status.battery = self.send_command('battery?', expectResponse = True)
            except Exception as ex:
                self.drone_status.battery = None
                logger.info({'action': 'get_drone_status',
                             'ex': ex})
            time.sleep(2)

    def reconnect(self):
        attempts = 0
        while attempts<3:
            attempts +=1
            logger.error(f'trying to reconnect')
            self.socket.close()
            time.sleep(0.3)
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.socket.bind((self.host_ip, self.host_port))
                logger.info('recconecting success')
                self.bindSuccess = True
                self.send_command('command', expectResponse=True)
                self.send_command('streamon', expectResponse=True)
                return
            except Exception as ex:
                logger.info(f'failed to reconnect {ex}')


    def receive_response(self, stop_event):
        while not stop_event.is_set():
            if (self.bindSuccess):
                try:
                    self.response, ip = self.socket.recvfrom(3000)
                    logger.info({'action': 'receive_response',
                                 'response': self.response})
                except socket.error as ex:
                    logger.error({'action': 'receive_response',
                                 'ex': ex})
                    #comeback to complere rebind attempt

    def __dell__(self):
        self.stop()

    def stop(self):
        self.stop_event.set()
        retry = 0
        while self._response_thread.is_alive():
            time.sleep(0.3)
            if retry > 30:
                break
            retry += 1
        self.socket.close()
        os.kill(self.proc.pid, 9)
        # Windows
        # import signal
        # os.kill(self.proc.pid, signal.CTRL_C_EVENT)

    def send_command_thread(self, command, blocking=True, expectResponse =True):
        self._command_thread = threading.Thread(
            target=self._send_command_async,
            args=(command, blocking,))
        self._command_thread.start()
        response = None
        if (expectResponse):
            retry = 0
            while self.response is None:
                time.sleep(0.01)
                if retry > 3:
                    break
                retry += 1

            if self.response is None:
                response = None
                self.bindSuccess = False
            else:
                response = self.response.decode('utf-8')
            self.response = None
        return response

    def send_command(self, command, blocking=True, expectResponse = False):
        '''
        is_acquire = self._command_semaphore.acquire(blocking=blocking)
        if is_acquire:
            with contextlib.ExitStack() as stack:
                stack.callback(self._command_semaphore.release)
        '''

        if (self.bindSuccess == False): self.reconnect()

        if (not self.bindSuccess):
            logger.info('reconnect failed')
            return None
        else:
            #logger.info({'action': 'send_command - sending', 'command': command})
            self.socket.sendto(command.encode('utf-8'), self.drone_address)
            response = None
            if expectResponse:
                retry = 0
                while self.response is None:
                    time.sleep(0.05)
                    if retry > 3:
                        break
                    retry += 1

            if self.response is None and expectResponse:
                response = None
                self.bindSuccess = False
            if self.response != None:
                response = self.response.decode('utf-8')
            self.response = None
            logger.info({'action': 'send_command - sent', 'command': command, 'response': response})
            return response

#else: logger.warning({'action': 'send_command', 'command': command, 'status': 'not_acquire'})

    def send_command_combined(self, command):
        #return "ok" #comeback
        attempts = 1
        while attempts<3:
            try:
                self.socket.sendto(command.encode('utf-8'), self.drone_address)
                time.sleep(0.01)
                response, ip = self.socket.recvfrom(3000)
                logger.info(f'command: {command}, response: {response}, attempts: {attempts}')
                return response
            except Exception as ex:
                attempts += 1
                logger.Error (f'faled to send command: {ex}')
                self.socket.close()
                time.sleep(0.03)
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.socket.bind((self.host_ip, self.host_port))
                logger.info('recconecting success')
                self.send_command('command')
                self.send_command('streamon')
        return "failed"



    def send_command_original(self, command):
        return 'ok' #comeback
        if (self.reconnecting): return self.response
        if (not self.bindSuccess):
            try:
                logger.info({'action': '_send_command', 'message': 'binding off - reconnecting'})
                self.reconnecting = True
                #self.socket.shutdown()
                self.socket.close()
                time.sleep(1)
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.socket.bind((self.host_ip, self.host_port))
                self.bindSuccess = True
                self.reconnecting = False
                logger.info('recconecting success')
                self.send_command('command')
                self.send_command('streamon')
            except Exception as ex:
                self.bindSuccess = False
                logger.info({'action':'send_command - ecconecting failure', 'ex':ex})
                self.reconnecting = False
                time.sleep(2)

        if (self.bindSuccess):
            logger.info({'action': 'send_command', 'command': command})
            try:
                self.socket.sendto(command.encode('utf-8'), self.drone_address)
            except Exception as ex:
                logger.error({'action': 'send_command', 'ex': ex})

            retry = 0
            while self.response is None:
                time.sleep(0.3)
                if retry > 3:
                    break
                retry += 1

            if self.response is None:
                response = None
                self.bindSuccess = False
            else:
                response = self.response.decode('utf-8')
            self.response = None
            return response

    def takeoff(self):
        return self.send_command('takeoff')

    def land(self):
        return self.send_command('land')

    def move(self, direction, distance):
        distance = float(distance)
        if self.is_imperial:
            distance = int(round(distance * 30.48))
        else:
            distance = int(round(distance * 100))
        return self.send_command(f'{direction} {distance}')

    def up(self, distance=DEFAULT_DISTANCE):
        return self.move('up', distance)

    def down(self, distance=DEFAULT_DISTANCE):
        return self.move('down', distance)

    def left(self, distance=DEFAULT_DISTANCE):
        return self.move('left', distance)

    def right(self, distance=DEFAULT_DISTANCE):
        return self.move('right', distance)

    def forward(self, distance=DEFAULT_DISTANCE):
        return self.move('forward', distance)

    def back(self, distance=DEFAULT_DISTANCE):
        return self.move('back', distance)

    def set_speed(self, speed):
        return self.send_command(f'speed {speed}')

    def clockwise(self, degree=DEFAULT_DEGREE):
        return self.send_command(f'cw {degree}')

    def counter_clockwise(self, degree=DEFAULT_DEGREE):
        return self.send_command(f'ccw {degree}')

    def flip_front(self):
        return self.send_command('flip f')

    def flip_back(self):
        return self.send_command('flip b')

    def flip_left(self):
        return self.send_command('flip l')

    def flip_right(self):
        return self.send_command('flip r')

    def patrol(self):
        if not self.is_patrol:
            self.patrol_event = threading.Event()
            self._thread_patrol = threading.Thread(
                target=self._patrol,
                args=(self._patrol_semaphore, self.patrol_event,))
            self._thread_patrol.start()
            self.is_patrol = True

    def stop_patrol(self):
        if self.is_patrol:
            self.patrol_event.set()
            retry = 0
            while self._thread_patrol.isAlive():
                time.sleep(0.3)
                if retry > 300:
                    break
                retry += 1
            self.is_patrol = False

    def _patrol(self, semaphore, stop_event):
        is_acquire = semaphore.acquire(blocking=False)
        if is_acquire:
            logger.info({'action': '_patrol', 'status': 'acquire'})
            with contextlib.ExitStack() as stack:
                stack.callback(semaphore.release)
                status = 0
                while not stop_event.is_set():
                    status += 1
                    if status == 1:
                        self.up()
                    if status == 2:
                        self.clockwise(90)
                    if status == 3:
                        self.down()
                    if status == 4:
                        status = 0
                    time.sleep(5)
        else:
            logger.warning({'action': '_patrol', 'status': 'not_acquire'})

    def receive_drone_video(self, stop_event, pipe_in, host_ip, video_port):
        while True:
            try:
                logger.info({'action': 'receive__drone_video', 'info': 'trying to establish video connection'})
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock_video:
                    sock_video.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock_video.settimeout(.5)
                    sock_video.bind((host_ip, video_port))
                    data = bytearray(2048)
                    errorCounter = 0
                    success = True
                    while (not stop_event.is_set()) and success:
                        try:
                            size, addr = sock_video.recvfrom_into(data)
                            errorCounter = 0
                            # logger.info({'action': 'receive__drone_video', 'data': data})
                        except socket.timeout as ex:
                            logger.warning({'action': 'receive__drone_video', 'ex': ex })
                            if (errorCounter == 3):
                                success = False
                            else:
                                errorCounter = errorCounter +1
                                time.sleep(0.5)
                                continue
                        except socket.error as ex:
                            logger.error({'action': 'receive__drone_video', 'ex': ex})
                            success = False
                        if (not success):
                            logger.error({'action': 'video connection failed'})
                            break

                        try:
                            pipe_in.write(data[:size])
                            pipe_in.flush()
                        except Exception as ex:
                            logger.error({'action': 'receive__drone_video', 'ex': ex})
                            break

            except Exception as ex2:
                logger.error({'action': 'receive__drone_video', 'ex': ex2})
                time.sleep(2)


    def video_binary_generator(self):
        while True:
            try:
                frame = self.proc_stdout.read(FRAME_SIZE)
            except Exception as ex:
                logger.error({'action': 'video_binary_generator', 'ex': ex})
                continue

            if not frame:
                continue

            frame = np.fromstring(frame, np.uint8).reshape(FRAME_Y, FRAME_X, 3)
           # cv.imshow('video', frame)
            yield frame

    def video_binary_generator_file (self, cap):
        logger.info('getting video from file/camera')
        # cap = cv.VideoCapture(0)

        # Check if camera opened successfully
        if (cap.isOpened() == False):
            logger.error("Error opening video stream or file")
        processLive = True
        # Read until video is completed
        target = {}
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                cv.imshow('original', frame)
                yield frame

    def enable_face_detect(self):
        self._is_enable_face_detect = True

    def disable_face_detect(self):
        self._is_enable_face_detect = False

    def video_jpeg_generator(self):
        logger.info('video processing method')
        while True:
            while not self.frameUpdated:
                continue
            self.frameUpdated = False
            frame =self.lastFrame
            _, jpeg = cv.imencode('.jpg', frame)
            jpeg_binary = jpeg.tobytes()

            if self.is_snapshot:
                backup_file = time.strftime("%Y%m%d-%H%M%S") + '.jpg'
                snapshot_file = 'snapshot.jpg'
                for filename in (backup_file, snapshot_file):
                    file_path = os.path.join(
                        SNAPSHOT_IMAGE_FOLDER, filename)
                    with open(file_path, 'wb') as f:
                        f.write(jpeg_binary)
                self.is_snapshot = False
            yield jpeg_binary


    def video_processing(self, stop_event):
        #pic = cv.imread('droneapp/testing/images/patio.jpg')
        #cv.imshow('pic',pic)
        #cv.waitKey(0)
        #out = cv.VideoWriter('output.avi', -1, 20.0, (FRAME_X, FRAME_Y))
        while not stop_event.is_set():
            """
            cap = cv.VideoCapture(0)
            #cap = cv.VideoCapture('droneapp/testing/images/flight2.mp4')
            while (cap.isOpened()):
                # Capture frame-by-frame
                ret, frame = cap.read()
                if ret == True:
                    """
            while True:
                for frame in self.video_binary_generator():

                    if (self.track_target): self.tracker.get_target(frame)
                    cv.putText(frame, self.mission_state.message, (50, 200), cv.FONT_HERSHEY_COMPLEX, 0.8, (150, 0, 0), 1)
                    cv.putText(frame, self.drone_status.battery if self.drone_status.battery != None else '--',
                               (FRAME_X-75, 50), cv.FONT_HERSHEY_COMPLEX, 1, (150, 0, 0), 1)
                    self.lastFrame = frame
                    self.frameUpdated = True
                    cv.imshow('original', self.lastFrame)
                    cv.waitKey(1)
                    self.frameUpdatedForMissinManager = True
                    #time.sleep(0.01)
                    #out.write(frame)
                    if (self.tracker.debug):
                        while True:
                            if cv.waitKey(1) & 0xFF == ord('z'):
                                break

                    #if cv.waitKey(1) & 0xFF == ord('q'):
                        #out.release()

            cv.waitKey(0)
          #  for frame in self.video_binary_generator_file(cap):
          #      cv.imshow('video', frame)

    def manage_mission(self, stop_event):
        pythoncom.CoInitialize()
        speaker = wincl.Dispatch('SAPI.SpVoice')
        speaker.speak('start')
        newState = True
        while not stop_event.is_set():
            time.sleep(0.01)
            if (self.mission_state.standby):
                self.show_state_message('standby - waiting for binding')
                if {self.bindSuccess}:

                    self.mission_state.standby = False
                    self.mission_state.vertical_take_off = True
                    newState = True


            elif(self.mission_state.vertical_take_off):
                if newState:
                    self.show_state_message('vertical take off')
                    speaker.speak('taking off in')
                    wait = 5
                    while wait > 0:
                        self.show_state_message(f'standby - taking off in {wait} sec')
                        speaker.speak(str(wait))
                        time.sleep(0.5)
                        wait -= 1
                        retries = 0
                    newState=False
                if not self.mission_state.test:
                    response = self.send_command('takeoff')
                    time.sleep(3)
                    self.send_command('up 50')
                if (self.mission_state.test or (self.bindSuccess and response is None)):
                    self.mission_state.vertical_take_off = False
                    self.mission_state.dock = True
                    newState = True
                else:
                    if retries==3:
                        self.show_state_message(f'vertical take off - error: {response}')
                        self.mission_state.vertical_take_off = False
                        self.mission_state.failed = True
                        newState = True
                time.sleep(0.3)
                retries += 1

            elif (self.mission_state.dock):
                newState = False
                docking_state = DockingState(start = True)
                self.tracker = Tracker(False)
                if self.dock(docking_state, speaker, self.mission_state.test):
                    self.mission_state.dock = False
                    self.mission_state.charge = True
                    newState = True
                else:
                    self.mission_state.dock = False
                    self.mission_state.failed = True
                    newState = True

            elif (self.mission_state.charge):
                if newState:
                    self.show_state_message('charging')
                    speaker.speak('charging')
                    newState = False

            elif self.mission_state.failed:
                self.show_state_message('failed - landing')
                speaker.speak('failed - landing')
                if not self.mission_state.test:  self.send_command('land')
                break

            elif self.mission_state.vertical_landing:
                self.show_state_message('landing')
                self.send_command('land')
                break


            elif self.mission_state.follow_me:
                if newState:
                    self.track_target = True
                    message = ('following you' + ' - test' if self.mission_state.test else '')
                    self.show_state_message(message)
                    speaker.speak(message)
                    newState=False
                self.move_to_range(destinationRange=6, pk=-0.75, destinationAccuaracy=5, maxCmd=50, test = self.mission_state.test, callerMessage='follow me', endByRangeOnly=True)

            elif self.mission_state.test_servos:
                self.control_servos(b'0 45 90 180')
                self.mission_state.test_servos = False



    def dock(self, docking_state, speaker, test):
        newState = True
        while (self.mission_state.dock):
            time.sleep(0.01)

            if (docking_state.start):
                self.show_state_message('docking - start')
                docking_state.start = False
                docking_state.approach = True

            elif (docking_state.approach):
                self.show_state_message('docking - approaching')
                if True:
                    docking_state.approach = False
                    docking_state.search_dock = True
                else:
                    self.show_state_message('docking-fail to send command')
                    docking_state.approach = False
                    docking_state.failed = True

            elif (docking_state.search_dock and self.frameUpdatedForMissinManager):
                self.frameUpdatedForMissinManager = False
                if newState:
                    self.show_state_message('searching')
                    speaker.speak('searching target')
                    self.track_target = True
                    scan_start_time = int(cv.getTickCount()/cv.getTickFrequency())
                    max_scan_duration = 8
                    decsending_speed = -20
                    self.show_state_message(f'searching: speed: {decsending_speed}% for {max_scan_duration}sec')
                    if (not (self.tracker.targetValid)):    # self.tracker.predictedTarget['center'][1] > FRAME_Y/2)):
                        cmd = f'rc {0} {0} {decsending_speed} {0}'
                        if not test: self.send_command(cmd)
                        newState = False
                if (self.tracker.targetValid) :  #and self.tracker.predictedTarget['center'][1] > FRAME_Y/2):
                    if not test: self.send_command('rc 0 0 0 0')
                    self.show_state_message('found')
                    speaker.speak('found')
                    docking_state.search_dock = False
                    docking_state.get_to_run_base = True
                    newState = True
                else:
                    scan_duration = int(cv.getTickCount()/cv.getTickFrequency()) - scan_start_time
                    if (scan_duration > max_scan_duration):
                        self.send_command('rc 0 0 0 0')
                        self.show_state_message('searching time out')
                        speaker.speak('target was not found')
                        docking_state.search_dock = False
                        docking_state.failed = True
                        newState = True


            elif (docking_state.get_to_run_base and self.frameUpdatedForMissinManager) :
                self.frameUpdatedForMissinManager = False
                if newState:
                    self.track_target = True
                    self.show_state_message('going to run base')
                    speaker.speak('going to run base')
                    newState = False
                if (self.move_to_range(destinationRange=120, pk=-0.5, destinationAccuaracy=5, maxCmd=50, test = test, callerMessage='going to run base', endByRangeOnly=False)):
                        docking_state.get_to_run_base = False
                        docking_state.docking_run = True
                        self.lastCmd = None
                        self.show_state_message('got to run base')
                        speaker.speak('got to run base')
                        newState = True


            elif (docking_state.docking_run and self.frameUpdatedForMissinManager):
                self.frameUpdatedForMissinManager = False
                if newState:
                    self.show_state_message('starting docking run')
                    speaker.speak('starting docking run')
                    self.track_target = True
                    newState = False
                self.move_to_range(destinationRange=-30, pk=-0.5, destinationAccuaracy=10, maxCmd=30, test = test, callerMessage='final run', endByRangeOnly=True)
                if self.tracker.predictedTarget['range'] < 28 and self.tracker.targetValid:
                    docking_state.docking_run = False
                    docking_state.flip = True
                    self.lastCmd = None
                    self.track_target = False
                    newState=True



            elif (docking_state.flip):
                cmd = 'flip f'
                if not test: self.send_command(cmd)
                self.lastCmd = cmd
                docking_state.flip = False
                docking_state.end = True
                self.show_state_message('flipping')
                speaker.speak('flipping')
                return True

            elif (docking_state.failed):
                self.show_state_message('docking failed')
                speaker.speak('docking failed')
                return False


    def move_to_range(self, destinationRange, pk, destinationAccuaracy, maxCmd, test, callerMessage, endByRangeOnly):
            if (not self.tracker.targetValid):
                self.show_state_message(f'{callerMessage} - waiting for lock')
                yaw = 0
                x = 0
                y = 0
                z = 0
            else:
                yaw = self.correction_cmd(
                    value_range=FRAME_X,
                    current= self.tracker.predictedTarget['center'][0],
                    desired=FRAME_X/2, p=-1, tolerance = 3, max_cmd = 50)

                x = self.correction_cmd(
                    value_range=FRAME_X,
                    current=self.tracker.predictedTarget['center'][0],
                    desired=FRAME_X / 2, p=-0.5, tolerance=5, max_cmd=50)
                '''
                x = self.correction_cmd(
                    value_range= 0.4,
                    current= self.tracker.predictedTarget['sideAngle'],
                    desired=1, p= 0.25, tolerance=10, max_cmd = 20)
                '''

                y = self.correction_cmd(
                    value_range=FRAME_Y,
                    current= self.tracker.predictedTarget['center'][1],
                    desired=FRAME_Y/5, p=0.7, tolerance=5, max_cmd = 50)

                z = self.correction_cmd(
                    value_range=300,
                    current=self.tracker.predictedTarget['range'],
                    desired=destinationRange, p=pk, tolerance=5, max_cmd = maxCmd)

            cmd = f'rc {0} {z} {y} {yaw}'
            #cmd = f'rc {x} {z} {y} {0}'
            if (cmd != self.lastCmd):
                self.show_state_message(f'{callerMessage}: {cmd}')
                if (not test):  response = self.send_command(cmd)
                self.lastCmd = cmd
                logger.info('command was sent')

            if (self.tracker.targetValid and
                    (abs(z) < destinationAccuaracy and (endByRangeOnly or (yaw ==0 and abs(x) < destinationAccuaracy and abs(y)<destinationAccuaracy)))):
                self.show_state_message('drone in destination')
                return True

            else: return False






    def correction_cmd (self, value_range, current, desired, p, tolerance, max_cmd = 100):
        err = 100*(2*(desired - current)/value_range)
        if (abs(err)  > tolerance):
            correction = int(err * p)
            if (correction > max_cmd): correction = max_cmd
            if (correction < -max_cmd) : correction = -max_cmd
            return correction
        else: return 0

    def show_state_message (self, message):
        logger.info(message)
        self.mission_state.message = message


    def gps_positioning(self):
        if(True):
            self.auto_land_stage = 'visual_search'

    def search_station(self):
        command = 'cw 2'
        logger.info(f'searching: {command})')
        # self.send_command(f'go {drone_x} {drone_y} {drone_z} {speed}',
        #                 blocking=False)


    def auto_land (self, target):


        diff_x = FRAME_CENTER_X - target['center'][0]
        diff_y = FRAME_CENTER_Y - target['center'][1]
        face_area = target['area']
        percent_face = face_area / FRAME_AREA

        drone_x, drone_y, drone_z, speed = 0, 0, 0, self.speed
        if diff_x < -30:
            drone_y = -30
        if diff_x > 30:
            drone_y = 30
        if diff_y < -15:
            drone_z = -30
        if diff_y > 15:
            drone_z = 30
        if percent_face > 0.30:
            drone_x = -30
        if percent_face < 0.02:
            drone_x = 30

        command = f'go {drone_x} {drone_y} {drone_z} {speed}'
        logger.info(command)
        #self.send_command(f'go {drone_x} {drone_y} {drone_z} {speed}',
         #                 blocking=False)


    def _send_command_async(self, command, blocking=True):
        is_acquire = self._command_semaphore.acquire(blocking=blocking)

    def video_processing_old(self, stop_event):
        while not stop_event.is_set():
            logger.info ('starting video processing method')
            for frame in self.video_binary_generator_file():
                self.lastFrame = frame
                self.frameUpdated = True
                logger.info(('got new frame'))
                if self._is_enable_face_detect:
                    if self.is_patrol:
                        self.stop_patrol()

                    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                    for (x, y, w, h) in faces:
                        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                        face_center_x = x + (w/2)
                        face_center_y = y + (h/2)
                        diff_x = FRAME_CENTER_X - face_center_x
                        diff_y = FRAME_CENTER_Y - face_center_y
                        face_area = w * h
                        percent_face = face_area / FRAME_AREA

                        drone_x, drone_y, drone_z, speed = 0, 0, 0, self.speed
                        if diff_x < -30:
                            drone_y = -30
                        if diff_x > 30:
                            drone_y = 30
                        if diff_y < -15:
                            drone_z = -30
                        if diff_y > 15:
                            drone_z = 30
                        if percent_face > 0.30:
                            drone_x = -30
                        if percent_face < 0.02:
                            drone_x = 30
                        self.send_command(f'go {drone_x} {drone_y} {drone_z} {speed}',
                                          blocking=False)
                        break

                cv.putText(frame, self.battery, (5,5), cv.FONT_HERSHEY_COMPLEX, 1, 0, 2)
                cv.imshow('video', frame)
                yield frame


    def snapshot(self):
        self.is_snapshot = True
        retry = 0
        while retry < 3:
            if not self.is_snapshot:
                return True
            time.sleep(0.1)
            retry += 1
        return False

