
# Age and Gender Prediction using ResNet-50

## Project Overview
This project aims to predict the **age** and **gender** of individuals in images using a **ResNet-50 deep learning model**. The model is fine-tuned on the **UTKFace dataset**, which contains images of faces along with their corresponding age and gender labels.

The **ResNet-50** model, which is pre-trained on the **ImageNet dataset**, is used to extract features from face images. We modify the architecture to include an age prediction and a gender classification head and fine-tune it on the **UTKFace dataset** for age and gender prediction.

## Dataset
The dataset used for this project is the **UTKFace dataset**, which contains face images labeled with age, gender, and race. For this project, we focus on **age** and **gender prediction**.

### Dataset Features:
- **Age**: Age of the individual in the image (in years).
- **Gender**: Gender of the individual (Male or Female).
- **Image**: RGB images of face crops (256 x 256 pixels).

You can download the dataset from [Kaggle - UTKFace Dataset](https://www.kaggle.com/datasets/jangedoo/utkface-new).
![image](https://github.com/user-attachments/assets/c96b3d3f-a9fc-43a5-8ba0-c1c19c66b35b)


## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- OpenCV
- NumPy
- Matplotlib
- Tkinter
- Pillow
- scikit-learn
- Kaggle dataset (`utkface-new`)

### Install dependencies:
```bash
pip install tensorflow opencv-python numpy matplotlib pillow tkinter scikit-learn
```

## Project Structure
```bash
|-- model.h5       # Not included here as size is 115mb and github not allowed  # to be generated when you use jupiter notebook  # Pre-trained model (ResNet-50 with custom layers)
|-- gui.py                   # Main script with Tkinter GUI for image upload and real-time predictions
|-- README.md                # This file
|--pred-age-gender-multioutput-keras-functional-api.ipynb     #kaggle notebook for finetuning rensnet50 with UTK dataset
```
kaggle notebook link : (https://www.kaggle.com/code/ankitpandey1576/pred-age-gender-multioutput-keras-functional-api/edit)
## How to Use

### 1. Real-Time Video Capture:
- Press the "Real-time" button to start the webcam.
- The model will continuously predict age and gender in real-time as you face the camera.

### 2. Upload Image for Prediction:
- Press the "Upload an Image" button to upload an image.
- The model will predict the gender and age based on the uploaded image and display the result.

### Example Usage:
![image](https://github.com/user-attachments/assets/96d39f6a-3757-42fb-9f20-cafa35a409ed)

- **Real-Time**: The camera will open, and the model will predict the gender and age in real-time.
  - The predictions will be displayed on the camera feed.
  - Press 'q' to exit the real-time feed.
  - ![image](https://github.com/user-attachments/assets/bd94a66a-5766-478b-9071-318a092ceef5)

- **Upload Image**: Select an image using the file dialog, and the model will predict the gender and age, showing the image with the predicted results.
- ![Screenshot 2024-10-17 143744](https://github.com/user-attachments/assets/ce6b6791-216b-4653-b460-741174df56d2)


## Model Architecture

The model uses **ResNet-50** as the base network and adds custom layers on top for **age** and **gender prediction**. The network is structured as follows:

1. **Base Model**: Pre-trained ResNet-50 without the top classification layers (weights from ImageNet).
2. **Global Average Pooling**: Reduces the spatial dimensions.
3. **Fully Connected Layers**: Dense layers for age and gender prediction.
4. **Age Output**: A linear activation function to predict the continuous age value.
5. **Gender Output**: A softmax activation function for binary classification (Male or Female).

### Example Output:
- **Gender Prediction**: "Male" or "Female"
- **Age Prediction**: The model will predict an age range (e.g., 25-29 years).

## Model Fine-Tuning

The **ResNet-50** model is fine-tuned on the **UTKFace dataset** for both age and gender prediction. The fine-tuning process involves the following steps:

1. **Load Pre-trained ResNet-50**: The base model, **ResNet-50**, is loaded with weights pre-trained on ImageNet without the top layers.
2. **Freeze Layers**: Most of the layers of the ResNet-50 model are frozen to retain their learned features.
3. **Custom Heads**:
   - A **Global Average Pooling** layer is added to reduce the output dimensions.
   - Two custom dense layers are added:
     - **Age Output**: A dense layer with one neuron and a **linear activation function** to predict the continuous age value.
     - **Gender Output**: A dense layer with two neurons and a **softmax activation function** for binary classification (Male or Female).
4. **Compile and Train**:
   - The model is compiled with a suitable optimizer and loss functions for age (MSE) and gender (categorical crossentropy).
   - The model is trained on the **UTKFace dataset** with **transfer learning** by unfreezing the last few layers of ResNet-50 and fine-tuning them to the age and gender prediction task.
   
The following script fine-tunes the model:
```python
# Load the pre-trained ResNet-50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze all layers of ResNet-50
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers for age and gender prediction
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
age_output = Dense(1, activation='linear', name='age')(x)
gender_output = Dense(2, activation='softmax', name='gender')(x)

# Define the full model
model = Model(inputs=base_model.input, outputs=[age_output, gender_output])

# Compile the model
model.compile(optimizer='adam', 
              loss={'age': 'mean_squared_error', 'gender': 'categorical_crossentropy'},
              metrics=['accuracy'])

# Train the model on the UTKFace dataset
model.fit(train_data, [age_labels, gender_labels], epochs=10, batch_size=32)
```

## Code Walkthrough

### 1. Data Preprocessing:
- The input image is resized and normalized to match the model's expected input shape.
- Grayscale images are converted to RGB if necessary to ensure compatibility with the ResNet-50 model.

### 2. Model Training:
- The ResNet-50 model is fine-tuned on the UTKFace dataset with age and gender labels.
- The model is trained using **transfer learning**, where the base layers of ResNet-50 are frozen, and only the top layers are trained.

### 3. Real-Time Prediction:
- **OpenCV** is used to capture frames from the camera feed.
- The frames are preprocessed, and predictions are made using the fine-tuned ResNet-50 model.

### 4. Image Upload for Prediction:
- **Tkinter** is used to create a GUI for image uploading.
- The uploaded image is processed and predictions are displayed using **matplotlib**.

## Results

After fine-tuning the **ResNet-50** model, it achieved accurate age and gender predictions on the test set. The model predicts the **age range** and **gender** with high accuracy.
At epoch 30 these are the acuracy measures 
![image](https://github.com/user-attachments/assets/34e7562b-19e1-485e-b92d-a1125a75eb1c)


## 🔗 Links
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ankitpandey1576/)
[![kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)](https://www.kaggle.com/ankitpandey1576)
