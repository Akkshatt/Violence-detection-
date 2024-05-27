
from flask import Flask, render_template, request
import os

app = Flask(__name__)

import cv2
import numpy as np
from keras.models import load_model
import sys

# Load the pre-trained VGG16 model for feature extraction
from keras.applications import VGG16
from keras.models import Model

image_model = VGG16(include_top=True, weights='imagenet')
transfer_layer = image_model.get_layer('fc2')
image_model_transfer = Model(inputs=image_model.input, outputs=transfer_layer.output)


model = load_model('VDM.h5')


def preprocess_video(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
     
        transfer_values = image_model_transfer.predict(np.expand_dims(frame, axis=0))
        frames.append(transfer_values)
    cap.release()
    return np.array(frames)


def predict_violence(video_path):
    frames = preprocess_video(video_path)
    predictions = model.predict(frames)
   
    violence_probability = np.mean(predictions[:, 0]) 
    return violence_probability

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index2.html', error='No file part')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index2.html', error='No selected file')

        if file:
            video_path = os.path.join('uploads', file.filename)
            file.save(video_path)
            violence_probability = predict_violence(video_path)
            os.remove(video_path)
            return render_template('index2.html', result={'probability': violence_probability, 'file_name': file.filename})
    
    return render_template('index2.html')

if __name__ == '__main__':
    app.run(debug=True)
