# Car Plate Detection using TensorFlow
This repository contains code for detecting and recognizing car license plates using TensorFlow. The project involves training a model to detect car plates in images and performing Optical Character Recognition (OCR) to extract the text from the detected plates.

# Requirements
- Python 3.x
- TensorFlow
- OpenCV
- Numpy
- Matplotlib
- Pytesseract
- TensorFlow Hub
  
# Dataset
The dataset used for this project is sourced from `Kaggle`, contains images of cars along with corresponding annotations in XML format specifying the bounding boxes around the license plates.

# Data Loading and Preprocessing
The `load_data` function reads the images and annotations from the dataset folders and preprocesses the data for training. The images are resized to 224x224 pixels, and the bounding box coordinates are normalized.

# Model Architecture
The model architecture is based on transfer learning using the `MobileNetV2` pre-trained model as a base. The final layers include a Global Average Pooling layer followed by Dense layers for regression to predict the bounding box coordinates. The model is compiled using the `Adam optimizer` and `Mean Squared Error loss function`.

# Training
The training data is split into training and validation sets using a train-test split of 80-20%. The model is trained for `200 epochs` with a `batch size of 32`.

# Prediction and OCR
After training, the `predict_bounding_boxes` function can be used to predict the bounding boxes for car plates in new images. The perform_ocr function extracts text from the predicted bounding boxes using Pytesseract for OCR.

# Usage
To use the provided code:
- Ensure all necessary dependencies are installed.
- Set the dataset path and directories for images and annotations.
- Load the data using load_data.
- Train the model using model.fit.
- Predict bounding boxes on new images using predict_bounding_boxes and perform OCR using perform_ocr.

# Results
The performance of the model is evaluated based on Mean Absolute Error (MAE) and accuracy metrics during training. Sample predictions are demonstrated with visualizations showing the detected plates and recognized text.

