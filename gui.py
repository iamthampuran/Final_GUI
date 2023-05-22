import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import cv2
from skimage.feature import hog
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from skimage import exposure
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as pi

import os
import numpy as np


base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(64, 64, 3))

# Load the VGG16 model (pre-trained on ImageNet)
net_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Load the VGG16 model for preprocessing
vgg_model = VGG16(weights='imagenet', include_top=False)

amd_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


glaucoma_model = load_model("./../Glaucoma/sanjgla3.h5")
print("Glaucoma model imported")
dr_model = load_model("./../Diabetic Retinopathy/Diabetic.h5")
print("DR Model imported")
amd_model = load_model("./../AMD/amd2.h5")
print("AMD Model imported")
myopia_model = load_model("./../Myopia/new-myopia3.h5")
print("Myopia model imported")
hypertension_model = load_model("./../Hypertension/hyper4.h5")
print("Hypertension model imported")

def check_glaucoma(image_path):
    class_names = {0: 'Glaucoma Free', 1: 'Presence of glaucoma'}
    # Load the image and process it with OpenCV
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (256, 256))  # resize the image to 256x256
    hog_features = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys")
    
    # Use the model to make a prediction
    new_features = np.array([hog_features])
    predictions = glaucoma_model.predict(np.expand_dims(new_features, axis=-1))
    
    print("Glaucoma predicted")
    # Get the class name of the predicted class
    class_idx = np.argmax(predictions[0])
    if predictions[0] > 0.5:
        class_idx = 1
    else:
        class_idx = 0
    print("Id = ", class_idx, "Predictions = ", predictions[0], "Classnames = ", class_names)
    print("Class = ", class_names[class_idx])
    class_name = class_names[class_idx]
    return class_name

def check_dr(img_path):
    # Create a dictionary to map the class indices to their names
    class_names = {0: 'No DR', 1: 'DR'}
     # Load the image and process it with OpenCV
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_not(thresh)
    resized = cv2.resize(thresh, (224, 224))
    color = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    img_array = np.array(color)
    img_array = img_array / 255.0
    
    # Use the model to make a prediction
    predictions = dr_model.predict(np.expand_dims(img_array, axis=0))
    print("DR predicted")
    # Get the class name of the predicted class
    class_idx = np.argmax(predictions[0])
    if predictions[0] > 0.5:
        class_idx = 1
    else:
        class_idx = 0
    print("Id = ",class_idx,"Predictions = ",predictions[0],"Classnames = ",class_names)
    class_name = class_names[class_idx]
    return class_name


def check_amd(img_path):
    # Create a dictionary to map the class indices to their names
    class_names = {0: 'No AMD', 1: 'AMD'}
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = amd_vgg.predict(x)
    feature.flatten()
     # Use the model to make a prediction
    predictions = amd_model.predict(np.expand_dims(feature, axis=0))
    print("AMD predicted")
    # Get the class name of the predicted class
    class_idx = np.argmax(predictions[0])
    if predictions[0] > 0.1:
        class_idx = 1
    else:
        class_idx = 0
    print("Id = ",class_idx,"Predictions = ",predictions[0],"Classnames = ",class_names)
    class_name = class_names[class_idx]
    return class_name

def check_myopia(img_path):
    # Create a dictionary to map the class indices to their names
    class_names = {0: 'No Myopia', 1: 'Presence of Myopia'}
    img = cv2.imread(img_path)
    
    img = cv2.resize(img, (224, 224))
    x = img_to_array(img)
    

    x = np.expand_dims(x, axis=0)
   

    x = preprocess_input(x, data_format='channels_last')
    imgfeatures = vgg_model.predict(x)
    imgfeatures = imgfeatures.flatten()

    # Use the model to make a prediction
    new_features = np.array([imgfeatures])
    predictions = myopia_model.predict(np.expand_dims(new_features, axis=-1))
    print("Myopia predicted")
    # Get the class name of the predicted class
    class_idx = np.argmax(predictions[0])
    if predictions[0] >= 0.5:
        class_idx = 1
    else:
        class_idx = 0
    print("Id = ", class_idx, "Predictions = ", predictions[0], "Classnames = ", class_names)
    print("Class = ", class_names[class_idx])
    class_name = class_names[class_idx]
    return class_name

def check_hypertension(img_path):
    # Create a dictionary to map the class indices to their names
    class_names = {0: 'No Hypertension', 1: 'Hypertension'}
    # Load the image and process it with OpenCV
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    # Split the image into RGB channels
    b, g, r = cv2.split(img)

    # Create a green plane image by replacing red and blue channels with zeros
    green_plane = cv2.merge((np.zeros_like(b), g, np.zeros_like(r)))

    gray = cv2.cvtColor(green_plane, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to segment the light reflex
    _, threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Invert the threshold image to create a mask
    mask = cv2.bitwise_not(threshold)

    # Bitwise AND the original image and the mask to remove the light reflex
    result = cv2.bitwise_and(green_plane, green_plane, mask=mask)

    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)

    # Split the LAB image into L, A, and B channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on the L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # Merge the processed L channel with the original A and B channels
    lab_processed = cv2.merge((l, a, b))

    # Convert the LAB processed image back to BGR color space
    result = cv2.cvtColor(lab_processed, cv2.COLOR_LAB2BGR)
    # Extract features using MobileNetV2
    feature = net_model.predict(np.expand_dims(result, axis=0))

    img_array = np.array(feature)
    
    # Use the model to make a prediction
    predictions = hypertension_model.predict(np.expand_dims(img_array, axis=0))
    print("Hypertension predicted")
    # Get the class name of the predicted class
    class_idx = np.argmax(predictions[0])
    if predictions[0] > 0.108:
        class_idx = 1
    else:
        class_idx = 0
    print("Id = ",class_idx,"Predictions = ",predictions[0],"Classnames = ",class_names)
    class_name = class_names[class_idx]
    return class_name


def open_image_file():
    # Open a file dialog to select an image file
    file_path = filedialog.askopenfilename()
    if file_path:
        # Classify the selected image
        amd_class_name = check_amd(file_path)
        hypertension_class_name = check_hypertension(file_path)
        gluacoma_class_name = check_glaucoma(file_path)
        dr_class_name = check_dr(file_path)
        myopia_class_name = check_myopia(file_path)
        print(file_path)
        # Update the GUI to display the selected image and its classification result
        img = Image.open(file_path)
        img = img.resize((400, 300))
        img = ImageTk.PhotoImage(img)
        image_label.configure(image=img)
        image_label.image = img
        
        # Update the classification labels with their respective results
        name_label.configure(text="Name: "+os.path.basename(file_path),font=('Arial', 20, 'bold'), fg='red')
        amd_label.configure(text='AMD: ' + amd_class_name, font=('Arial', 20, 'bold'), fg='red')
        hypertension_label.configure(text='Hypertension: ' + hypertension_class_name, font=('Arial', 20, 'bold'), fg='red')
        glaucoma_label.configure(text='Glaucoma: ' + gluacoma_class_name, font=('Arial', 20, 'bold'), fg='red')
        dr_label.configure(text='DR: ' + dr_class_name, font=('Arial', 20, 'bold'), fg='red')
        myopia_label.configure(text='Myopia: ' + myopia_class_name, font=('Arial', 20, 'bold'), fg='red')
        header_label.configure(text='Disease Detection', font=('Arial', 25, 'bold'), fg='blue')


# Create the GUI
root = tk.Tk()
root.title('Eye Classification')
root.geometry('1280x720')

# Load the background image
background_image = Image.open('gradient.jpg')
background_image = background_image.resize((1280, 720))
background_photo = ImageTk.PhotoImage(background_image)

# Create a label with the background image
background_label = tk.Label(root, image=background_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)


# Add a header label
header_label = tk.Label(root, text='Disease Detection', font=('Arial', 24, 'bold'), fg='blue')
header_label.pack(pady=20)

# Create a button to open an image file dialog
button_image = tk.PhotoImage(file='button.png')
button = tk.Button(root, image=button_image, command=open_image_file, bg='white')
button.pack(pady=10)

# Create a label to display the selected image
image_label = tk.Label(root)
image_label.pack()

# Create labels to display the classification results

name_label = tk.Label(root, text='Name: ', font=('Arial', 20, 'bold'), fg='red',bg= root["bg"],bd=0)
name_label.pack()
amd_label = tk.Label(root, text='AMD: ', font=('Arial', 20, 'bold'), fg='red',bg= root["bg"],bd=0)
amd_label.pack()
hypertension_label = tk.Label(root, text='Hypertension: ', font=('Arial', 20, 'bold'), fg='red',bg= root["bg"],bd=0)
hypertension_label.pack()
glaucoma_label = tk.Label(root, text='Glaucoma: ', font=('Arial', 20, 'bold'), fg='red',bg= root["bg"],bd=0)
glaucoma_label.pack()
dr_label = tk.Label(root, text='DR: ', font=('Arial', 20, 'bold'), fg='red',bg= root["bg"],bd=0)
dr_label.pack()
myopia_label = tk.Label(root, text='Myopia: ', font=('Arial', 20, 'bold'), fg='red',bg= root["bg"],bd=0)
myopia_label.pack()

root.mainloop()