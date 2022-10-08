"""
Modules used:
    numpy, opencv-python, opencv-utils, tensorflow
"""

# all important imports
import numpy as np
from cv2 import cv2 as cv
from time import sleep

from tensorflow.keras import models

# defining model - load directly or define and then load weights
model = models.load_model('./model')
# printing model summary
print('\n\n')
print(model.summary())

# setting up the model output for opencv
outputToEmotionMap = {
    0: 'angry',
    1: 'disgusted',
    2: 'in fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprised'
}

outputToEmojiMap = {
    0: cv.imread('./emojis/angry.png'),
    1: cv.imread('./emojis/disgust.jpg'),
    2: cv.imread('./emojis/fear.png'),
    3: cv.imread('./emojis/happy.png'),
    4: cv.imread('./emojis/neutral.jpg'),
    5: cv.imread('./emojis/sad.png'),
    6: cv.imread('./emojis/surprise.png')
}


vidCapture = cv.VideoCapture(0)

# helper functions
def getFirstFaceBoundingClientRect(image):
    grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    faceCascade = cv.CascadeClassifier('haarcascade.xml')
    faces = faceCascade.detectMultiScale(
        grayImage,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )

    # returning only the first face
    try:
        return faces[0]
    except:
        return []


def drawRectangleOnImage(image, coordinates, rectColor=(0, 255, 0), rectThickness=2):
    x = coordinates[0]
    y = coordinates[1]
    w = coordinates[2]
    h = coordinates[3]
    return cv.rectangle(image, (x, y), (x+w, y+h), rectColor, rectThickness)


# starting video capturing
print('Starting Video Capture')

# some constants
messageCellPadding = 5
messageFontSize = 16

# flags to check the phase
runModel = False
outputDisplayed = True

while(1):
    ret, frame = vidCapture.read()
    message = 'No face recorded'
    messageCoords = [int(frame.shape[0] * 0.48), int(frame.shape[1] * 0.7)]

    # operations on the captured frame
    faceCoords = getFirstFaceBoundingClientRect(frame)
    # check if a face is there
    if len(faceCoords) > 0:
        frame = drawRectangleOnImage(frame, faceCoords)
        message = 'Press "r" to get emotion from face'
        messageCoords[0] = int(frame.shape[0] * 0.2)

        # if user is pressing r => send through model
        if cv.waitKey(1) & 0xFF == ord('r'):
            runModel = True
            # converting frame to model input
            x, y, w, h = faceCoords
            inputImage = np.expand_dims(np.expand_dims(cv.resize(cv.cvtColor(
                frame[y:y+h, x:x+w], cv.COLOR_BGR2GRAY), (48, 48)
            ), -1), 0)
            # predicting with help of model - taking one with max probability
            predictions = model.predict(inputImage)
            maxProbabilityPredict = np.argmax(predictions)

            # adjusting message
            message = 'You are {}'.format(
                outputToEmotionMap[maxProbabilityPredict]
            )
            messageCoords[0] = int(frame.shape[0] * 0.45)


            # adding emoji on face via circular masking
            maskForFrame = np.zeros(frame.shape[:2], dtype='uint8') + 255
            maskForFrame = cv.circle(maskForFrame, (int(x + w/2), int(y + h/2)), int((w+h)/4), 0, -1)
            
            maskForEmoji = cv.bitwise_not(maskForFrame)

            emojiPicture = outputToEmojiMap[maxProbabilityPredict]
            emojiPicture = cv.resize(emojiPicture, (h, w))
            duplicateFrame = np.copy(frame)
            duplicateFrame[y:y+h, x:x+w] = emojiPicture
            
            maskedImage = cv.bitwise_and(frame, frame, mask= maskForFrame)
            maskedEmoji = cv.bitwise_and(duplicateFrame, duplicateFrame, mask= maskForEmoji)

            frame = maskedEmoji + maskedImage


            


    # adding the message
    frame = drawRectangleOnImage(
        frame, (0, messageCoords[1] - messageCellPadding - messageFontSize,
                frame.shape[1], messageFontSize + 2 * messageCellPadding), (0, 0, 0), -1
    )
    frame = cv.putText(
        frame, message, tuple(messageCoords),
        cv.FONT_HERSHEY_COMPLEX_SMALL,
        1, (255, 255, 255), 1, cv.LINE_AA
    )

    # displaying the frame
    cv.imshow('Emojify', frame)
    if runModel:
        if not outputDisplayed: 
            sleep(5)
            runModel = False
            outputDisplayed = True
        else:
            outputDisplayed = False

    # ending the frame if q is pressed
    if cv.waitKey(5) & 0xFF == ord('q'):
        break

# finishing up the program
vidCapture.release()
cv.destroyAllWindows()

print('\n\n\nExiting program...\nThank you for using Emojify')
