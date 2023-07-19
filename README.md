# Hand_Gesture_Recognition

The hand gesture recognition model is trained on the data collected by user. The data needed for training can be easily changed by he user.

The code is written to recognize 3 distinct gestures and it can also be easily changed by the user to recognize any desired number of gestures.

The model uses Mediapipe to identify and get the positions (x and y coordinates) of hand landmarks. The classification machine learning model is then used to train the model on this data. 

The classification model used here is RandomForestClassifier from Scikit-learn. The accuracy achieved with this model is 100%.

The data_collection.py file would open the camera and you get the instruction to start capturing pictures of you. You need to position your hand like the gesture you want. This process will continue with number of gestures you desire. The code will create a file named Collected_Data which would contain the images of gestures in their respective files.

The data_processing.py file extracts the relevant information from these images and stores it in a file named training_data.pickle.

The training.py file trains the model on this data and stores it in the file named model.pickle.

The model is now ready!

We can test the model by running the test.py file.
