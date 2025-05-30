from flask import Flask, render_template, Response
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

facetrack = tf.keras.models.load_model('model/facetracker.h5')

app = Flask(__name__)
cap = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = cap.read()

        if not success:
            break
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = tf.image.resize(rgb, (120,120))
        
        yhat = facetrack.predict(np.expand_dims(resized/255,0))
        sample_coords = yhat[1][0]
        
        if yhat[0] > 0.5: 
            image_width, image_height = 640, 480
            x_center, y_center, box_width, box_height = sample_coords 
            
            x_center_pixel = x_center * image_width
            y_center_pixel = y_center * image_height
            width_pixel = box_width * image_width
            height_pixel = box_height * image_height
            
            # Compute xmin, ymin, xmax, ymax
            x_min = int(x_center_pixel - width_pixel / 2)
            y_min = int(y_center_pixel - height_pixel / 2)
            x_max = int(x_center_pixel + width_pixel / 2)
            y_max = int(y_center_pixel + height_pixel / 2)
            # Controls the main rectangle
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), 
                                (0,255,0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield(b'--frame\r\n'
              b'content-type: image/jepg\r\n\r\n' + frame +
              b'\r\n')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug = True)