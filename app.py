from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Use a non-interactive backend

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('mnist_model.keras')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        img = Image.open(io.BytesIO(file.read()))
        img = img.convert('L')  # Convert to grayscale
        img = img.resize((28, 28))  # Resize to 28x28
        img = np.array(img)
        img = img / 255.0  # Normalize
        img = img.reshape(1, 28, 28)  # Reshape to match model input shape

        # Display the image using matplotlib
        plt.imshow(img.reshape(28, 28), cmap='gray')
        plt.savefig('static/images/uploaded_image.png')  # Save the image to a file

        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)
        return render_template('result.html', prediction=predicted_class, image_path='static/images/uploaded_image.png')


if __name__ == '__main__':
    app.run(debug=True)
