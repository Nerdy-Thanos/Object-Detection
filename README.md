# Object-Detection
Work project. Object detection model program to detect objects present in a picture and extract features to make predictions.

object detection script to detect features like
  ->color
  
  ->size
  
  ->shape
  
  ->depth
  
  ->Pixel Density

feature map generators help extract these features and load them into the model

model.pb file is the actual object detection model

labels file contains the labels or the objects the model is trained to detect

predict.py is the script that calls the object detection script to get the features

an image is loaded into the model and the objects present into it are detected and stored into a csv file.
