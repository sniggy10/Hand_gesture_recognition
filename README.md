# Hand_gesture_recognition

Final Year Project created to classify hand gestures using CNN and predict real-time results with OpenCv

### Objective
The aim is to build a hand gesture recognition model with deep learning. This model will classify images of different hand gestures, such as a fist, palm, thumb, and others. 
This model could be used for communicating with the deaf and dumb using these various gestures.
This will help to make the communication between the deaf/dumb and the people who don’t understand sign language a lot easier.
It can also be used in applications involving gesture navigation.

## Steps to Run :
Clone the repository

### Prerequisites
Install the follwing packages
1. Numpy
2. OpenCv
3. Tensorflow
4. Keras

### Training the model
1. Run `python collect-data.py` to create the data folder with training and test images
2. Run `python train.py` to train the model

### Real Time Prediction
Run `python live-prediction.py` for prediction on real time data