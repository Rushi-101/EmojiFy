# Emojify - Finding your emotions
The goal of our project is to classify human expressions into 7 different classes such as angry, fear, happy, sad etc. and make emojis out of these expressions.

In this project, we classified human facial expressions using deep learning (CNN) and filter and map corresponding emojis or avatars using OpenCV.

The images are trained and tested on the FER2013 dataset comprising of 7 classes of human emotions namely angry, disgust, fear, happy, neutral, sad, surprise with around 28K training images and 7K testing images.

## How to run
To run the project, extract Program.zip and open up a terminal/powershell/command prompt. Inside the terminal, go to the directory where the files are extracted. 
Then run the following command in the terminal:
* In Windows: py main.py
* In Linux: python3 main.py

The python installation must have the following packages:
* numpy ()
* opencv-python
* tensorflow

We have used openCV to develop the UI of our project. The interface is pretty simple. Directions on how to use the program is as follows:

1. Run the program (main.py). Wait for the program to open the camera’s video feed.
2. If no face is detected, the program will remain as it is.
3. If a face is detected, click ‘r’ to take a picture. The picture is then sent through model. The emotion is then displayed on screen. 

Refer to the presentation for a video demonstration. 

Contribution of the members:
1. Rushikesh = Developing the CNN with Saksham - Basic Structure of CNN
2. Saksham = Developing the CNN with Rushikesh - Optimization of parameters
3. Shubham = Implementing the UI of the program, Implementation of CNN with help of Tensorflow
Common Work:
Research and Pre planning	
