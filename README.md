# Pet Identifying Convolutional Neural Network Model

## **Overview**
The purpose of this project is to create a Deep Learning Convolutional Neural Network image classifier model that can identify various types of pets from an uploaded .jpg or .png image file. This model should correctly identify if a given image is of a cat, dog, bird, fish, hamster or bearded dragon.

This image classification model was developed with TensorFlow, and was trained and tested using thousands of images of cats, dogs, birds, fish, hamsters and bearded dragons scraped from Flickr and Imgur using Beautiful Soup. The model was then tested and optimized locally, before being deployed on AWS EC2. Finally, a webpage was created to allow users to upload images of pets which are fed through the model via API created using Flask.

The following steps have been outlined in more detail below:
1. [Collection and processing of the image files for the various pet types](#data-extraction-and-processing) 
2. [Creation and optimization of the Convolutional Neural Network (CNN) model on TensorFlow](#creation-and-optimization-of-the-convolutional-neural-network-model)
3. [Deployment of the model for end-user interaction](#deployment-of-the-model-for-end-user-interaction)

## **Data Extraction and Processing**
Images to train and test the CNN model to recognize the different pet types were collected from Flickr and Imgur, then converted to png files and reduced to a size of 240x240 pixels.

### **Image Extraction from Flickr and Imgur**
Images were extracted using the Beautiful Soup webscraping library. A 'for' loop was created to collect 500 images from either Flickr or Imgur. For Flickr, the following URL was appended with a search term to collect various images of cats, dogs, birds, fish, hamsters and bearded dragons: "https://www.flickr.com/search/?text=". For Imgur, the following URL was appended with search terms: "https://imgur.com/search?q=".

When extracting images, there were a few considerations made to improve model accuracy:
- The images were of singular animals, rather than groups
- Various orientations of the animals were collected
- Any images where the animal is difficult to see were scrapped

```python
import os
import requests
from bs4 import BeautifulSoup

# What image are you looking for
search_query = "one bird"

# What directory you want update this
save_directory = "../Bird Images"
os.makedirs(save_directory, exist_ok=True)

# Update the number of images to download.
num_images_to_download = 1000

# base URL for image search. Comment out/Comment in the one you wish to use. So far works on 2 sites. Imgur uses png, Flickr uses jpg.
# base_url = "https://imgur.com/search?q="
base_url = "https://www.flickr.com/search/?text="

# Initialize counters
total_images_downloaded = 0
page_number = 1

while total_images_downloaded < num_images_to_download:
    # Construct the search URL for the current page
    search_url = base_url + search_query + "&page=" + str(page_number)

    # Send a GET request to the search URL
    response = requests.get(search_url)

    # Parse the HTML content of the response using BeautifulSoup
    soup = BeautifulSoup(response.content, "html.parser")

    # Find the image elements in the HTML
    image_elements = soup.find_all("img")

    for img in image_elements:
        # Get the image source URL
        image_url = img.get("src")

        # Skip if the image source URL is not valid
        if not image_url or image_url.startswith("data:") or not (image_url.endswith('.jpg') or image_url.endswith('.png') or image_url.endswith('.gif')):
            continue

        # Add the scheme if it's missing
        if image_url.startswith("//"):
            image_url = "https:" + image_url

        try:
            # Send a GET request to download the image
            image_response = requests.get(image_url)

            # Skip if the image response status is not 200
            if image_response.status_code != 200:
                print(f"Skipping image: {image_url} (status code: {image_response.status_code})")
                continue

            # Save the image to the specified directory
            image_path = os.path.join(save_directory, f"image_{total_images_downloaded + 1}.jpg")
            with open(image_path, "wb") as f:
                f.write(image_response.content)

            print(f"Downloaded image: {image_path}")

            total_images_downloaded += 1

            if total_images_downloaded >= num_images_to_download:
                break

        except Exception as e:
            print(f"Error occurred while downloading image: {image_url}")
            print(str(e))

    page_number += 1

print("Complete")
```
Extracting images that met the above criteria required an iterative process of manually reviewing the images and scrapping the ones that were not needed. Several search terms were used to find more relevant data. E.g. for birds, "small bird", "pet bird", and "bird" were all used to get a comprehensive dataset. We also noticed that Flickr was much more accurate at pulling images using the given search terms than Imgur.

Results from Flickr:
![Flickr Results](/Images/Flickr_vs_imgur/Flickr.png "Results from Flickr")

Results from Imgur:
![Imgur Results](/Images/Flickr_vs_imgur/Imgur.png "Results from Imgur")

Flickr produced more releavant results for training the model than Imgur.

The data extraction code can be found in the following file: webscraperforimages.ipynb 

### **Image Filetype Conversion, Resizing and Renaming**
After collecting the required images, the dataset was cleaned up to establish a consistent file format and image size for the model to train on. A for loop was created to parse through each image using Pillow and CV2 to convert any .jpg or .jpeg filetypes into .png, as well as resize these images to 240x240 pixels. 

Having the same filetype and image size was important for the model to compare different types of animals. However, one important consideration was to maintain the aspect ratio of each image to ensure better recognizability of the animal. This was done by reducing the image size while maintaining the ratio of its height and width to fit within a 240x240 pixel frame. 

Here is the function used to create a clear background for an image resized to fit into a 240x240 pixel frame while maintaining the aspect ratio:

```python
def resize_image(input_image_path, output_image_path, size):
    original_image = Image.open(input_image_path)
    original_image.thumbnail(size, Image.ANTIALIAS)

    # Create a new square background image
    background = Image.new('RGBA', size, (255, 255, 255, 0))

    # Calculate the position to paste the thumbnail
    image_position = (int((size[0] - original_image.size[0]) / 2), int((size[1] - original_image.size[1]) / 2))

    # Paste the thumbnail onto the background
    background.paste(original_image, image_position)

    # Save the output image
    background.save(output_image_path)
```

All the image files extracted from Flickr and Imgur were processed by applying this function through a for loop and storing the processed file in the same directory while deleting the unprocessed file with the same name. Finally, the processed images were renamed to start with the name of the respective animal and the image number: e.g. dog1.png

More details of the image file conversion and resizing can be found in: resizer240240.ipynb
More details of the image file renaming can be found in: rename_to_image_nr.ipynb

## **Creation and Optimization of the Convolutional Neural Network Model**
The processed image files for each animal were randomly separated into test and train data directories, with 25% of images randomly sorted into a test folder and the remaining 75% of images being sorted into the training folder. A Convolutional Neural Network Model was created using Tensor Flow, and tested on Google Colab for faster testing. The model accuracy was evaluated and the model was optimized by adjusting the number of epochs, increasing the dataset, and changing the number of nodes and neural layers.

### **Separating the Testing and Training Datasets**
First the directories for the test and train images for each animal need to be created with the following structure:
..
    Test
        bird
        cat
        dog
        dragon
        fish
        hamster
    Train
        bird
        cat
        dog
        dragon
        fish
        hamster
    
```python
from os import makedirs
# create directories
dataset_home = ''
subdirs = ['train/', 'test/']
for subdir in subdirs:
 # create label subdirectories
    labeldirs = ['dog/', 'cat/', 'fish/', 'bird/', 'hamster/', 'dragon/']
    for labldir in labeldirs:
        newdir = dataset_home + subdir + labldir
        makedirs(newdir, exist_ok=True)
```

Then, by fixing the seed value of the pseudoradnom number generator value (using the random library), 25% can be held back everytime the following code is run:

```python
from shutil import copyfile
from random import seed
from random import random

# seed random number generator
imgdirs = ['dog', 'cat', 'fish', 'bird', 'hamster', 'dragon']
labeldirs =['dog/', 'cat/', 'fish/', 'bird/', 'hamster/', 'dragon/']

seed(1)
# define ratio of pictures to use for validation
val_ratio = 0.25
# copy training dataset images into subdirectories
for imgdir, label in zip(imgdirs, labeldirs):
    src_directory = f'../Images/{imgdir}'
    for file in listdir(src_directory):
        src = src_directory + '/' + file
        dst_dir = 'train/'
        if random() < val_ratio:
            dst_dir = 'test/'
        if file.startswith(imgdir):
            dst = dataset_home + dst_dir + label + file
            copyfile(src, dst)
```
More info on how the test and train datasets were split can be found in the test_train_directories.ipynb file in the Test_train_sets directory.

### **Defining and Training the Model**
The Convolutional Neural Network (CNN) model was constructed and trained using Tensorflow. Here are the libaries and dependies used to define, train and export the model to an h5 file:

```python
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

The model function was defined with a specific convolutional layer based architecture similar to that defined in the following paper: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)The models architecture was designed by stacking convolutional layers with 3x3 filters followed by a max pooling layer. The depth of each layer was increased exponentially with the input layer having 32 nodes, the hidden layers having 64, 128 and 512 nodes respectively. Each of these layers was equiped with the 'ReLu' activation function.

The output layer was defined based on the number of different pets the model was instructed to predict, which equalled 6 nodes. This layer was then equiped with the softmax activation function as seen below:

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')  # 6 classes of animals
])
```

All images in the training and test directories were then processed using the ImageDataGenerator library from Tensorflow to be fed into the machine learning model. This scales the pixel values to the range of 0-1. The ```flow_from_directory()``` function was used to iterate through the test and training directories separately. The class_mode was defined as "categorical" as the machine learning problem requires category recognition across more than two categories. The batch size was fixed at 32.

```python
# Image properties
IMG_HEIGHT, IMG_WIDTH = 240, 240
BATCH_SIZE = 32

# Preprocess and augment the training data using ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

# Just rescale the test data
test_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical')

# Flow validation images in batches using test_datagen generator
test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                  batch_size=BATCH_SIZE,
                                                  class_mode='categorical')
```

The model was then compiled and trained across 30 epochs to yield an accuracy rate of roughly 89% on the final epoch. Even with 30 epochs, running this model on the free version of Google Colab, on processing speeds of 2.20 GHz with 13 GB RAM caused issues with keeping the machine on for too long. This model had to be trained locally using the Tensorflow library installed on a local machine. 

```python
# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# Train the model
model.fit(train_generator, validation_data=test_generator, epochs=30)
```

Model Results:
```python
Epoch 100/100
129/129 [==============================] - 141s 1s/step - loss: 0.2733 - accuracy: 0.8951 - val_loss: 0.6956 - val_accuracy: 0.7963
```

Finally the model was exported in h5 format for deployment:
```python
# Define the path to save the model in your Google Drive
save_path = '/content/drive/MyDrive/animal_classifier.h5'
# Save the model to the specified path in your Google Drive
model.save(save_path)
```

## **Deployment of the model for end-user interaction**
A flask API framework was created to allow uploaded images to be stored in a temporary folder, preprocessed to fit the dimmensions and filetype of the model (i.e. .png filetype and image frame of 240x240 pixels). The large model file as well as uploaded images were stored on EC2 on AWS and routed through the flask API for the model to make a prediction on newly uploaded images. A webpage was then set up where images of pets can be uploaded by end-users and the model's predictions are displayed. 

### **Creating the Flask API framework**
Flask was set up to receive image files in various formats and sizes, and process these using Pillow to create .png image files that fit the 240x240 pixel frame the model is used to. Then, these images were fed through the model using load_model and image libraries in Tensorflow.

```python
from flask import Flask, request, render_template # redirect, url_for
import os
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
```

The flask API relies on three functions:
1. A function to process the uploaded image to meet the criteria of the model
- Opens the image from temporary path
- Converts to RGBA
- resizes to 240X240 via thumbnail function to preserve aspect ratio
- Saves as PNG
```python
#Image preprocessing function
def preprocess_image(file_path):
    # Open the image file
    img = Image.open(file_path)
    
    # Convert to RGBA
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    
    # Resize the image to 240x240 pixels
    img.thumbnail((240,240))
    
    # Save the image as PNG format
    png_path = os.path.splitext(file_path)[0] + ".png"
    img.save(png_path, "PNG")
    
    # Return the path to the converted image
    return png_path
```

2. A function to produce an array of the image, and feed it through the model to produce a prediction
- Opens the preprocessed image
- Converts the image to an array
- Uses Tensorflow Keras "model.predict"
- Returns the results as one of the pet class labels

```python
def classify_image(file_path):
    img = image.load_img(file_path, target_size=(240, 240))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    class_label = class_labels[class_index]

    return class_label
```

3. A function to receive the uploaded image via the API and run it through the above two functions
- Main function starts by returning the upload.html file (via render_template function). This is technically a "GET" request to the flask API.
- "If" the user uploads an image and clicks "upload", this puts in a POST request to flask.
- The image is uploaded to a temporary location
- With a POST request, the image will undergo image processing and image classifying, returning the result on the result.html page

```python
def upload_file():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return 'No file part in the request'

        file = request.files['file']

        # Check if a file was uploaded
        if file.filename == '':
            return 'No file selected'

        #Save the file to the temporary storage directory
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        #Perform image Preprocessing (resizing, format change)
        preprocessed_image = preprocess_image(file_path) 

        #Perform image classification
        result = classify_image(preprocessed_image)

        #Delete the temporary uploaded file
        os.remove(file_path)
        
        if os.path.exists(preprocessed_image):
                os.remove(preprocessed_image)

        return render_template('result.html', result=result)

    return render_template('upload.html')
```

The details of the flask set up code can be found in the app.py file in this repository.

### **Deploying the Model on EC2**

### **Creating the HTML Page for End-Users and Connecting to Flask**
