from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.models import Model

app = Flask(__name__)

# Load the trained LSTM model
model_path = "C:/Users/Lenovo/Violence-Detection-CNN-LSTM/VDM2.h5"
model = load_model(model_path)

def preprocess_image(image):
    # Resize the image to match the input dimensions of VGG16
    image = cv2.resize(image, (224, 224))
    # Preprocess the image for VGG16 model
    image = preprocess_input(image)
    return image

def extract_transfer_values(image):
    # Load the pre-trained VGG16 model
    base_model = VGG16(weights='imagenet')
    # Remove the last layer (classification layer) from the VGG16 model
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
    # Preprocess the image
    image = preprocess_image(image)
    # Add a batch dimension to the image
    image = np.expand_dims(image, axis=0)
    # Get the transfer values by passing the image through the model
    transfer_values = model.predict(image)
    return transfer_values

def predict_violence(image):
    # Extract transfer values from the image
    transfer_values = extract_transfer_values(image)
    # Reshape transfer values to match the input shape expected by the LSTM model
    transfer_values = np.expand_dims(transfer_values, axis=0)
    # Make predictions using the LSTM model
    prediction = model.predict(transfer_values)
    return prediction

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    image_file = request.files['file']
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    prediction = predict_violence(image)

    return jsonify({'data': {'probability_violence': float(prediction[0][1])}})

if __name__ == "__main__":
    app.run(debug=True)