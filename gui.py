import tkinter as tk
from tkinter import filedialog
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load the pre-trained model
model = load_model("model.h5")

# Gender dictionary to map the prediction to label
gender_dict = {0: "Male", 1: "Female"}


# Preprocess the input image for prediction
def preprocess_image(image):
    # Resize the image to the input size the model expects (e.g., 128x128 or 200x200)
    img_resized = cv2.resize(image, (128, 128))

    # Convert to grayscale if the image has only one channel (if it's grayscale)
    if len(img_resized.shape) == 3 and img_resized.shape[-1] == 1:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB

    # If the image has only 1 channel, convert it to 3 channels
    if len(img_resized.shape) == 2 or img_resized.shape[-1] == 1:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB

    # Convert the image to an array and add an additional dimension to represent batch size
    img_array = img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize the image (ResNet50 expects images to be preprocessed)
    img_array = preprocess_input(img_array)

    return img_array


# Function to handle real-time video capturing
def start_realtime():
    # Initialize the camera
    camera = cv2.VideoCapture(0)

    while True:
        # Capture frame from the camera
        ret, frame = camera.read()

        # Preprocess the captured frame
        preprocessed_image = preprocess_image(frame)

        # Make predictions using the model
        predictions = model.predict(preprocessed_image)
        gender_prediction = gender_dict[round(predictions[1][0][0])]
        age_prediction = round(predictions[0][0][0])
        age_lower = age_prediction - 3
        age_upper = age_prediction + 3

        # Display the original frame and predicted gender/age
        cv2.putText(frame, f"Predicted Gender: {gender_prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                    2)
        cv2.putText(frame, f"Predicted Age: {age_lower} to {age_upper}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 2)
        cv2.imshow("Gender and Age Prediction", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(700) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    camera.release()
    cv2.destroyAllWindows()


# Function to preprocess image for upload
def upload_image():
    file_path = filedialog.askopenfilename()  # Opens a file dialog to select an image

    if file_path:
        # Open the image using PIL and convert it to a numpy array
        image = Image.open(file_path)
        image = np.array(image)

        # Preprocess the uploaded image
        preprocessed_image = preprocess_image(image)

        # Make predictions using the model
        predictions = model.predict(preprocessed_image)

        # Extract predictions for gender and age
        gender_prediction = gender_dict[round(predictions[1][0][0])]  # Gender prediction
        age_prediction = round(predictions[0][0][0])  # Age prediction

        # Calculate the age range
        age_lower = age_prediction - 3
        age_upper = age_prediction + 3

        # Display the original image and predicted gender/age range
        plt.imshow(image)
        plt.title(f"Predicted Gender: {gender_prediction}, Predicted Age: {age_lower} to {age_upper}")
        plt.axis("off")
        plt.show()


# Create the main window using Tkinter
window = tk.Tk()
window.geometry("800x600")

# Frame for the menu screen
menu_screen = tk.Frame(window)
menu_screen.pack()

# Button to start real-time video capture
realtime_button = tk.Button(menu_screen, text="Real-time", command=start_realtime)
realtime_button.pack(pady=10)

# Button to upload and predict an image
upload_button = tk.Button(menu_screen, text="Upload an Image", command=upload_image)
upload_button.pack(pady=10)

# Run the Tkinter event loop
window.mainloop()
