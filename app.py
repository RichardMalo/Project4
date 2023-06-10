from flask import Flask, request, render_template # redirect, url_for
import os
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Set the temporary image storage directory
UPLOAD_FOLDER = 'D:/Code/temp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained model
model = load_model('D:/DataBootcamp/Projects/Project4/Template/animal_classifier.h5')

# Define the class labels
class_labels = ['bird', 'cat', 'dog', 'dragon', 'fish', 'hamster']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return 'No file part in the request'

        file = request.files['file']

        # Check if a file was uploaded
        if file.filename == '':
            return 'No file selected'

        # Save the file to the temporary storage directory
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        preprocessed_image = preprocess_image(file_path) 

        # Perform image classification
        result = classify_image(preprocessed_image)

        # Delete the temporary uploaded file
        os.remove(file_path)
        
        if os.path.exists(preprocessed_image):
                os.remove(preprocessed_image)


        return render_template('result.html', result=result)

    return render_template('upload.html')

def classify_image(file_path):
    img = image.load_img(file_path, target_size=(240, 240))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    class_label = class_labels[class_index]

    return class_label

def preprocess_image(file_path):
    # Open the image file
    img = Image.open(file_path)
    
    # Convert to RGB if necessary
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    
    # Resize the image to 240x240 pixels
    img.thumbnail((240,240))
    
    # Save the image as PNG format
    png_path = os.path.splitext(file_path)[0] + ".png"
    img.save(png_path, "PNG")
    
    # Return the path to the converted image
    return png_path

if __name__ == '__main__':
    app.run()
