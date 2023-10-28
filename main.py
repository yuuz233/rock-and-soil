import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define the classes and their corresponding directories
classes = ['Sand', 'Clay', 'Sandy Loam', 'Loam']
directories = ['/data/Sand', '/data/Clay', '/data/Sandy Loam', '/data/Loam']

# Load the data
data = []
labels = []
for i, directory in enumerate(directories):
    for filename in os.listdir(directory):
        img = cv2.imread(os.path.join(directory, filename))
        img = cv2.resize(img, (128, 128))  # Resize the image to a fixed size
        data.append(img)
        labels.append(i)

# Convert the data and labels to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Normalize the pixel values to the range [0, 1]
data = data / 255.0

# Split the data into training and testing sets (you can adjust the ratio as needed)
split = int(data.shape[0] * 0.8)
train_data, test_data = data[:split], data[split:]
train_labels, test_labels = labels[:split], labels[split:]

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(classes), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, train_labels, epochs=10, batch_size=32)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(test_data, test_labels)
print('Test loss:', loss)
print('Test accuracy:', accuracy)
