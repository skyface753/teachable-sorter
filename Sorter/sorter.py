# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# this file will contain the main sorter class
# this will handle grabbing the flir images, determining

from pycoral.adapters import classify, common
from pycoral.utils.dataset import read_label_file

from pycoral.utils.edgetpu import make_interpreter
from utils import CameraWebsocketHandler
from utils.BiQuad import BiQuadFilter

from functools import partial
from PIL import Image
from scipy import ndimage
# import edgetpu.classification.engine
import threading
import asyncio
import base64
import utils
import cv2
import argparse
import sys
import numpy as np
import time
# try:
# import RPi.GPIO as GPIO
# GPIO.setwarnings(False)
# GPIO.setmode(GPIO.BOARD)
# GPIO.setup(sendPin, GPIO.OUT, initial=GPIO.LOW)
# except ImportError:
#     print("RPi.GPIO not found, skipping GPIO setup")
#     pass
# Path to edgetpu compatible model
model_path = '../model_edgetpu.tflite'

# NOTE: can either be 'train' to classify images using edgetpu or 'sort' to just send images to TM2
mode = "sort"
# sendPin = 7
filter_type = 'zone'
# biquad params : type, Fc, Q, peakGainDB
bq = BiQuadFilter('band', 0.1, 0.707, 0.0)


def send_over_ws(msg, cam_sockets):
    for ws in cam_sockets:
        # Send raw bytes instead of a string
        ws.write_message(msg, binary=True)


def format_img_tm2(cv_mat):
    ret, buf = cv2.imencode('.jpg', cv_mat)
    return buf.tobytes()  # Return raw binary data


interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()
labels = read_label_file('../labels.txt')  # Adjust path if needed


def classify_image(image):
    """Classifies an image using the Edge TPU model."""
    size = common.input_size(interpreter)
    image_numpy_array = np.array(image)
    common.set_input(interpreter, cv2.resize(image_numpy_array, size, fx=0, fy=0,
                                             interpolation=cv2.INTER_CUBIC))

    interpreter.invoke()
    return classify.get_classes(interpreter)

    # start_time = time.time()
    size = common.input_size(interpreter)
    image = image.convert('RGB').resize(size, Image.LANCZOS)

    # Preprocess the image
    params = common.input_details(interpreter, 'quantization_parameters')
    scale = params['scales']
    zero_point = params['zero_points']
    mean, std = 128.0, 128.0  # Default normalization values

    normalized_input = (np.asarray(image) - mean) / (std * scale) + zero_point
    np.clip(normalized_input, 0, 255, out=normalized_input)
    cv2.imshow('frame2', normalized_input.astype(np.uint8))
    common.set_input(interpreter, normalized_input.astype(np.uint8))

    # Run inference
    interpreter.invoke()
    results = classify.get_classes(interpreter, top_k=1, score_threshold=0.95)
    # end_time = time.time()
    # print(f"Time taken: {end_time - start_time:.4f}s")
    return results


# this is the logic that determines if there is a sorting target in the center of the frame
def is_good_photo(img, width, height, mean, sliding_window):
    detection_zone_height = 20
    detection_zone_interval = 5
    threshold = 4.5
    if (filter_type == 'zone'):
        detection_zone_avg = img[height // 2: (
            height // 2) + detection_zone_height: detection_zone_interval, 0:-1:3].mean()
    if (filter_type == 'biquad2d'):
        detection_zone_avg = abs(bq.process(img.mean))
    if (filter_type == 'biquad'):
        detection_zone_avg = abs(bq.process(
            img[height // 2: (height // 2) + detection_zone_height: detection_zone_interval, 0:-1:3].mean()))
    if (filter_type == 'center_of_mass'):
        center = scipy.ndimage.measurements.center_of_mass(img)
        detection_zone_avg = (center[0] + center[1]) / 2

    if len(sliding_window) > 30:
        mean[0] = utils.mean_arr(sliding_window)
        sliding_window.clear()

    else:
        sliding_window.append(detection_zone_avg)
    # print(detection_zone_avg)
    if mean[0] != None and abs(detection_zone_avg - mean[0]) > threshold:
        print("Target Detected Taking Picture")
        return True

    return False


# call each time you  have a new frame
def on_new_frame(cv_mat, mean, sliding_window, send_over_ws, cam_sockets, disable_servo=False, debug=False):
    img_pil = Image.fromarray(cv_mat)

    width, height = img_pil.size

    is_good_frame = is_good_photo(cv_mat, width, height, mean, sliding_window)
    if (is_good_frame):
        # NOTE: Teachable Machine 2 works on images of size 224x224 and will resize all inputs
        # to that size. so we have to make sure our edgetpu converted model is fed similar images.
        if (width, height) != (224, 224):
            img_pil.resize((224, 224))
        if debug:
            img = np.array(img_pil)
            cv2.imshow('good_frame', img)
            cv2.waitKey(1)
        if (mode == 'train'):
            # No need for dict, just send bytes
            message = format_img_tm2(cv_mat)
            send_over_ws(message, cam_sockets)
            # time.sleep(0.25) NOTE: debounce this at a rate depending on your singulation rate

        elif (mode == 'sort'):
            classification_result = classify_image(img_pil)
            if classification_result:
                label_id = classification_result[0].id
                confidence = classification_result[0].score
                label_name = labels.get(label_id, label_id)

                print(f"Detected: {label_name} ({confidence:.2f})")

                if label_id == 0 and confidence > 0.95:
                    # GPIO.output(sendPin, GPIO.HIGH)
                    # print("Coin 0")
                    if not disable_servo:
                        servo.min()
                else:
                    if not disable_servo:
                        servo.max()
                    # print("Coin 1")
                    # GPIO.output(sendPin, GPIO.LOW)


STOCK_IMAGE_2EURO = '../train_15_5_jpg.rf.fb3c716fe28c2375b8705b9ca89aa2b4.jpg'
STOCK_IMAGE_5CENT = '../IMG_4185_43_jpg.rf.c5d07f6f00965505d6271b0c1a981fb1.jpg'

SHOW_STOCK_IMAGE = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode_parser = parser.add_mutually_exclusive_group(required=False)
    mode_parser.add_argument('--train', dest='will_sort', action='store_false')
    mode_parser.add_argument('--sort', dest='will_sort', action='store_true')

    parser.add_argument('--debug', dest='debug', action='store_true')

    filter_parse = parser.add_mutually_exclusive_group(required=False)
    filter_parse.add_argument(
        '--zone-activation', dest='zone', action='store_true')
    filter_parse.add_argument('--biquad', dest='biquad', action='store_true')
    filter_parse.add_argument(
        '--biquad2d', dest='biquad2d', action='store_true')
    filter_parse.add_argument(
        '--center-of-mass', dest='center_of_mass', action='store_true')

    camera_parse = parser.add_mutually_exclusive_group(required=False)
    camera_parse.add_argument('--flir', dest='flir', action='store_true')
    camera_parse.add_argument('--opencv', dest='opencv', action='store_true')
    camera_parse.add_argument('--arducam', dest='arducam', action='store_true')

    parser.add_argument('--disable-servo',
                        dest='disable_servo', action='store_true')

    parser.set_defaults(will_sort=True)
    args = parser.parse_args()

    # Start the tornado websocket server
    cam_sockets = []
    new_loop = asyncio.new_event_loop()
    server_thread = threading.Thread(
        target=CameraWebsocketHandler.start_server, args=(new_loop, cam_sockets, ))
    server_thread.start()

    if args.will_sort:
        mode = "sort"
    else:
        mode = "train"

    if not args.disable_servo:
        from utils.servo import MyServo
        servo = MyServo(18)

    #  parse filter type
    if args.zone:
        filter_type = 'zone'
    elif args.biquad:
        filter_type = 'biquad'
    elif args.biquad2d:
        filter_type = 'biquad2d'
    elif args.center_of_mass:
        filter_type = 'center_of_mass'

    mean = [None]
    sliding_window = []

    if (args.flir):
        import FLIR
        print("Initializing Flir Camera")
        cam = FLIR.FlirBFS(on_new_frame=partial(on_new_frame, mean=mean, sliding_window=sliding_window,
                                                send_over_ws=send_over_ws, cam_sockets=cam_sockets, disable_servo=args.disable_servo),
                           display=args.debug, frame_rate=120)
        cam.run_cam()
    elif (args.arducam):
        raise Exception("Arducam Support Coming")
    else:
        cap = cv2.VideoCapture(1)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cv2_im = frame
            # read from file ../Sorter-Test/pycoral/coin/train_15_5_jpg.rf.fb3c716fe28c2375b8705b9ca89aa2b4.jpg
            if SHOW_STOCK_IMAGE:
                if SHOW_STOCK_IMAGE == 2:
                    cv2_im = cv2.imread(STOCK_IMAGE_2EURO)
                elif SHOW_STOCK_IMAGE == 5:
                    cv2_im = cv2.imread(STOCK_IMAGE_5CENT)
                else:
                    print("WHAT????")
            pil_im = Image.fromarray(cv2_im)
            pil_im.resize((224, 224))
            pil_im.transpose(Image.FLIP_LEFT_RIGHT)
            if args.debug:
                cv2.imshow('frame', cv2_im)
            on_new_frame(cv2_im, mean, sliding_window,
                         send_over_ws, cam_sockets, disable_servo=args.disable_servo)
            # if cv2.waitKey(1) & 0xff == ord('q'):
            #     break
            # stock image change by key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('2'):
                SHOW_STOCK_IMAGE = 2
            elif key == ord('5'):
                SHOW_STOCK_IMAGE = 5
            elif key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        print('Initializing opencv Video Stream')
