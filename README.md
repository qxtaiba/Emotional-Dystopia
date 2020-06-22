## Data

The data set used for training is the **Kaggle FER2013** emotion recognition data set : https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image. The task is to categorize each face based on the emotion shown in the facial expression in to one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).

## Requirements

```
Python : 3.6.5
Tensorflow : 1.10.1
Keras : 2.2.2
Numpy : 1.15.4
OpenCV : 4.0.0
```

## Files

The different files that can be found in this repo :
- `Notebooks` : Notebooks that have been used to obtain the final model
- `Model`: Contains the pre-trained models for emotion recognition, face detection and facial landmarks
- `Python` : The code to launch the live facial emotion recognition

Among the notebooks, the role of each notebook is the following :
- `01-Pre-Processing.ipynb` : Transform the initial CSV file into train and test data sets
- `02-HOG_Features.ipynb` : A manual extraction of features (Histograms of Oriented Gradients, Landmarks) and SVM
- `03-Pre-Processing-EmotionalDAN.ipynb` : An implementation of Deep Alignment Networks to extract features
- `04-LGBM.ipynb` : Use of classical Boosting techniques on top on flatenned image or auto-encoded image
- `05-Simple_Arch.ipynb` : A simple Deep Learning Architecture
- `06-Inception.ipynb` : An implementation of the Inception Architecture
- `07-Xception.ipynb` : An implementation of the Xception Architecture
- `08-DeXpression.ipynb` : An implementation of the DeXpression Architecture
- `09-Prediction.ipynb` : Live Webcam prediction of the model
- `10-Hybrid.ipynb` : A hybrid deep learning model taking both the HOG/Landmarks model and the image

## Video Processing

#### Pipeline

The video processing pipeline was built the following way :
- Launch the webcam
- Identify the face using a Histogram of Oriented Gradients
- Zoom in on the face
- Scale the face to 48 * 48 pixels
- Make a prediction on the face using a pre-trained model
- Identify the number of blinks on the facial landmarks on each picture

#### Model

The model we have chosen is an **XCeption** model. The model was tuned with:
- Data augmentation
- Early stopping
- Decreasing learning rate on plateau
- L2-Regularization
- Class weight balancing
- And kept the best model

To learn more about the **XCeption** model, check this article: https://maelfabien.github.io/deeplearning/xception/

## Predictors

The set of emotions we are trying to predict are the following:
- Happiness
- Sadness
- Fear
- Disgust
- Surprise
- Neutral
- Anger

## Sources

- Visualization: https://github.com/JostineHo/mememoji/blob/master/data_visualization.ipynb
- State of the art Architecture: https://github.com/amineHorseman/facial-expression-recognition-using-cnn
- Eyes Tracking: https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/
- Face Alignment: https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
- A Brief Review of Facial Emotion Recognition Based on Visual Information: https://www.mdpi.com/1424-8220/18/2/401/pdf
- Going deeper in facial expression recognition using deep neural networks: https://ieeexplore.ieee.org/document/7477450
- Emotional Deep Alignment Network paper: https://arxiv.org/abs/1810.10529
- Emotional Deep Alignment Network github: https://github.com/IvonaTau/emotionaldan
- HOG, Landmarks and SVM: https://github.com/amineHorseman/facial-expression-recognition-svm
