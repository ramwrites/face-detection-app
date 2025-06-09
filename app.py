from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import numpy as np
import cv2
import tensorflow as tf

facetrack = tf.keras.models.load_model('model/facetracker.h5')

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('image')
def handle_message(frame):
    image = np.frombuffer(frame, dtype= np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    resized = tf.image.resize(img, (120,120))
    
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
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), 
                            (0,255,0), 2)

    _, buffer = cv2.imencode('.jpg', img)
    socketio.emit('response', buffer.tobytes())


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=10000, debug=True, allow_unsafe_werkzeug=True)

