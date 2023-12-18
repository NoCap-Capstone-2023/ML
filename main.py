from flask import Flask, request, jsonify
import cv2
import numpy as np
import pickle
import tensorflow as tf

app = Flask(__name__)

# Load your model and class names here
# Replace 'loaded_model' and 'loaded_class_names' with your actual model and class names
# Example:
loaded_model = tf.keras.models.load_model('OCRmodel.h5')
with open('class_names.pkl', 'rb') as file:
    loaded_class_names = pickle.load(file)

@app.route('/predict', methods=['POST','GET'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'})

    image_file = request.files['image']
    nparr = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Apply morphological operations to clean up the image
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours based on x-coordinate to arrange left to right
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

    # List to store segmented regions
    segmented_regions = []

    # Loop through contours and extract bounding boxes for text regions
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Extract each detected text region
        text_roi = image[y:y + h, x:x + w]

        # Append the segmented region to the list
        segmented_regions.append(text_roi)

    # Assuming you have a pre-trained model named 'model' and a list of 'class_names' for your model's output classes
    # Predict the content of each segmented region
    predicted_text = ""

    for i, region in enumerate(segmented_regions):
        # Preprocess the segmented region (resize, normalization, etc.) - similar to your training data preprocessing
        # Example: Assuming the segmented region has been preprocessed and stored in 'segmented_regions_preprocessed'
        processed_region = cv2.resize(region, (64, 64))  # Resize the region to match your model input size
        processed_region = np.array(processed_region, dtype=np.float32)
        processed_region = processed_region / 255.0  # Normalize pixel values
        processed_region = processed_region.reshape(1, 64, 64, 3)  # Reshape to match model input shape

        # Make predictions using your pre-trained model
        predictions = loaded_model.predict(processed_region)

        # Get the predicted label for the region
        predicted_label_index = np.argmax(predictions, axis=1)
        predicted_label = loaded_class_names[predicted_label_index[0]]  # Assuming class_names contain label strings

        # Append the predicted label to form the predicted text
        predicted_text += predicted_label
        if i < len(segmented_regions) - 1:
            predicted_text += " "# Append the predicted label to form the predicted text

    # Print or use the predicted text
    # print("Predicted Text:", predicted_text)
    # Rest of your image processing code goes here...

    # Assuming the code to process and predict text is the same as in the provided code

    # Return the predicted text as JSON
    return predicted_text

@app.route('/')
def hello_world():
    return 'Hello World'
 

if __name__ == '__main__':
    app.run(debug=True)
