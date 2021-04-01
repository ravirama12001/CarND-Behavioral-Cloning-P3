import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


# Dividing raw data of images and steering, Throttle, break and speed
def get_data():
    images = []
    measurements = []
    lines = []
    correctionAngle = 0.2
    with open('SimulationData/driving_log.csv') as csvFile:
        reader = csv.reader(csvFile)
        next(reader)
        for line in reader:
            lines.append(line)
    for line in lines:
        # To Read all Center images
        currentPath = line[0].strip()
        img = cv2.imread(currentPath)
        images.append(img)
        measurements.append(float(line[3]))
            
        # To flip center image to augument data
        currentPath = line[0].strip()
        img = cv2.imread(currentPath)
        img = cv2.flip(img, 1)
        images.append(img)
        measurements.append(-1 * float(line[3]))
            
        # To Read all Left images
        currentPath =  line[1].strip()
        img = cv2.imread(currentPath)
        images.append(img)
        measurements.append(float(line[3]) + correctionAngle)
            
        # To flip left image to augument data
        currentPath = line[1].strip()
        img = cv2.imread(currentPath)
        img = cv2.flip(img, 1)
        images.append(img)
        measurements.append(-1 * (float(line[3]) + correctionAngle))
            
        # To read all Right images
        currentPath = line[2].strip()
        img = cv2.imread(currentPath)
        images.append(img)
        measurements.append(float(line[3]) - correctionAngle)
            
        # To flip center image to augument dataa
        currentPath = line[2].strip()
        img = cv2.imread(currentPath)
        img = cv2.flip(img, 1)
        images.append(img)
        measurements.append(-1 * (float(line[3]) - correctionAngle))

    return np.asarray(images), np.asarray(measurements).astype(np.float32)


# Data as Left, Center and Right images. and Lables as steering, Throttle, break and speed

data, labels = get_data()
from keras.models import Sequential
from keras.layers import Conv2D, Cropping2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D

# Build the model
activation = 'relu'
model = Sequential()
# Removing the unnecessary part from image like sky and mountains nearby
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
    
# Normalization
model.add(Lambda(lambda x: (x)/127. -1))
    
# Layer1: 24 filters of 5x5
model.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2), activation=activation))
    
#Layer2: 36 filters of 5x5
model.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2), activation=activation))
    
# Layer3: 48 filters of 5x5
model.add(Conv2D(filters=48, kernel_size=5, strides=(2, 2), activation=activation))
    
#Layer4: 64 filters of 3x3
model.add(Conv2D(filters=64, kernel_size=3, activation=activation))
    
#Layer5: 64 filters of 3x3
model.add(Conv2D(filters=64, kernel_size=3, activation=activation))
    
#Layer8: Flattening
model.add(Flatten())
    
#Layer7: 100 Fully connected
model.add(Dense(100))
    
#Layer8: 50 Fully connected
model.add(Dense(50))
    
#Layer9: 10 Fully connected
model.add(Dense(10))
    
#Layer10: 1 Fully connected
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.summary()

model.fit(data, labels, validation_split=0.2, shuffle=True, epochs=30)

print('Saved the model')

#Saving model as model.h5
model.save('model.h5')

#Plotting the model
model.