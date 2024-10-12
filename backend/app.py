from flask import Flask, request, jsonify
from flask_cors import CORS  # Ensure React frontend can make requests to Flask
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests from React

# Class names corresponding to the model's output
class_names = {
    0: "Burger",
    1: "Chicken Curry",
    2: "Fried Rice",
    3: "Donuts",
    4: "Dumplings",
    5: "French Fries",
    6: "Grilled Sandwich",
    7: "Ice Cream",
    8: "Pizza",
    9: "Samosa",
}

# Load the trained model
model = load_model('food_classification_model.h5')  # Update path to where the model is stored

# Define a function to predict the food class
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize the image to model's expected input size
    img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    preds = model.predict(img_array)  # Get predictions
    predicted_class = np.argmax(preds, axis=1)  # Get the index of the highest probability class
    return predicted_class[0]

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded image to a temporary location
    img_path = os.path.join('static', file.filename)
    file.save(img_path)

    # Make prediction
    predicted_class = predict_image(img_path)
    final_ans = class_names[predicted_class]

    return jsonify({'prediction': final_ans, 'image_file': file.filename})

@app.route('/')
def index():
    return "Food Classification API is running. Use '/predict' to upload an image."

if __name__ == '__main__':
    app.run(debug=True)
