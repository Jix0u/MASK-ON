# Loading Libraries
import cv2 as cv            
import mediapipe            
import numpy as np
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from flask import Flask, render_template, render_template_string, Response

app = Flask(__name__, template_folder='template')

capture = cv.VideoCapture(0)


def gen():    
    mask_detector = load_model("mask_detector.model")

    face_detector = mediapipe.solutions.face_detection.FaceDetection()


    while True:
        

        success, img = capture.read()

        abc = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
        results = face_detector.process(abc)
        if results.detections:
            for detection in results.detections:

                boxR = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape

                (startX, startY, endX, endY) = (boxR.xmin, boxR.ymin, boxR.width, boxR.height) * np.array([iw, ih, iw, ih])
                startX = max(0, int(startX))
                startY = max(0, int(startY))
                endX = min(iw - 1, int(startX + endX))
                endY = min(ih - 1, int(startY + endY))

                face = abc[startY:endY, startX:endX]
                face = cv.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.array([face], dtype='float32')

                preds = mask_detector.predict(face, batch_size=32)[0][0]
                label = "Mask" if preds < 0.5 else "No Mask"
                percentage = (1 - preds) * 100 if label == "Mask" else preds * 100
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                cv.putText(img, label, (startX, startY - 10), cv.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                cv.rectangle(img, (startX, startY), (endX, endY), color, 2)

        # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
        ret, buffer = cv.imencode('.jpg', img)
        img = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')  # concat frame one by one and show result

    else:
        capture.release()
@app.route('/')
def index():
    """Video streaming"""
    return render_template('videostuff.html')
@app.route('/video')
def video():
    """Video streaming"""
    return render_template('videostuff.html')

@app.route('/abc')
def abc():
    """Video streaming"""
    return render_template('abc.html')
@app.route('/ray')
def ray():
    """Video streaming"""
    return render_template('ray.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run()
