import os
import cv2 as cv
from ultralytics import YOLO

import numpy as np
import matplotlib.pyplot as pl

# OpenCV Setup
trainIteration = '11'
modelPath = os.getcwd() + '\\runs\\detect\\train' + trainIteration + '\\weights\\best.pt'
print("Using Model in Path: ", modelPath)

model = YOLO(modelPath)

cap = cv.VideoCapture('\\archive\\train\\0000f77c-6257be58.mov')

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    results = model.track(frame, stream=True)

    for result in results:
        class_names = result.names

        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()