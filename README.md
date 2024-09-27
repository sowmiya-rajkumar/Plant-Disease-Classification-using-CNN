# Plant-Disease-Classification-using-CNN
This project aims to classify potato leaf images into three categories — Early Blight, Late Blight, and Healthy, using a Convolutional Neural Network (CNN) implemented with TensorFlow and Keras.

## Dataset
The dataset is taken from Kaggle's Plant Village dataset and contains 2,152 images of potato leaves divided into the following classes:

Potato___Early_blight
Potato___Late_blight
Potato___Healthy
The images are of variable sizes and are processed into 256x256 resolution for uniformity during training.

## Requirements
To get started with the project, ensure you have the following dependencies installed:

TensorFlow >= 2.0
Keras
Matplotlib
NumPy
Python 3.x

## Data Preprocessing
The dataset is loaded using TensorFlow’s image_dataset_from_directory() method, which automatically assigns labels to each image based on its directory name. <br/ >
After loading, the dataset is split into training, validation, and testing sets with an 80/10/10 ratio.

## Model Architecture
The CNN model consists of several convolutional and max-pooling layers, followed by a dense layer to perform classification. Data augmentation (random flipping and rotation) is applied to improve the model's generalization.

Model Layers:
Rescaling and Resizing: Resize images to 256x256 and scale pixel values between 0 and 1.
Data Augmentation: Random flipping and rotation to reduce overfitting.
Convolutional Layers: Extract features using multiple Conv2D layers.
MaxPooling Layers: Downsample feature maps.
Dense Layers: Fully connected layers for classification.

## Training the Model
The model is compiled using the Adam optimizer and Sparse Categorical Crossentropy as the loss function. The model is trained for 50 epochs with a batch size of 32.

## Results
The model achieved strong performance on both the training and validation datasets. Below are sample accuracy results:

Training Accuracy: 99.31%
Validation Accuracy: 99.48%
Test Accuracy: (To be evaluated)

## Future Improvements
Hyperparameter Tuning: Adjust learning rate, batch size, and other hyperparameters to further improve accuracy.
Transfer Learning: Explore pre-trained models like VGG16 or ResNet for better feature extraction.
Model Deployment: Integrate the model into a web application for real-time predictions.
