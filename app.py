from flask import Flask, request, render_template, redirect, url_for
import os
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

        # Perform image classification
        result = classify_image(file_path)

        # Delete the temporary uploaded file
        os.remove(file_path)

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
    return class_index

if __name__ == '__main__':
    app.run()