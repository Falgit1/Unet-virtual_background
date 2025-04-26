# 1. Imports and Setup Code

Python
import tensorflow as tf
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm 
from PIL import Image
import matplotlib.pyplot as plt

## Explanation
The libraries used in the code:
TensorFlow: For building and training the segmentation model.
Pandas: For handling the dataset (CSV file).
NumPy: For numerical operations.
Pillow (PIL): For image processing.
Matplotlib: For visualization of images.
TQDM: For progress bars during data loading.

# 2. Loading Data from CSV
Code
Python
df = pd.read_csv("human-segmentation-main/train.csv")
df.head()

## Explanation
Reads the CSV file containing paths to training images and their corresponding masks.
The dataset contains two columns:
masks: Paths to ground-truth masks.
images: Paths to the images.

# 3. Visualizing an Image and its Mask
Code
Python
p1 = "human-segmentation-main/" + df["masks"][1]
img = Image.open(p1).resize((128, 128), Image.LANCZOS)
plt.imshow(img)
plt.show()

p1 = "human-segmentation-main/" + df["images"][1]
img2 = Image.open(p1).resize((128, 128), Image.LANCZOS)
plt.imshow(img2)
plt.show()

## Explanation
Loads an image and its corresponding mask for visualization.
Resizes both images to 128x128 using the LANCZOS filter.
Displays the images using Matplotlib.

# 4. Combining Images with Masks
Code
Python
img = np.array(img)
img2 = np.array(img2)
overlapped = []
shape = img2.shape

for i in range(shape[0]):
    l1 = []
    for j in range(shape[1]):
        if img[i][j] > 100:
            tem = img2[i][j]
        else:
            tem = img2[i][j] * 0
        l1.append(tem)
    overlapped.append(l1)

result = np.array(overlapped)
plt.imshow(result)
plt.show()

## Explanation
Converts the images and masks into numpy arrays.
Iterates through each pixel of the images and masks:
If the mask pixel value is above a threshold (100), the corresponding image pixel is retained.
Otherwise, the pixel is set to zero (black).
Displays the overlapped image.

# 5. Data Loading Function Code

Python
def load_data(data="train"):
    if data == "train":
        path = "images"
        count = df[path].count()
        train = np.zeros((count, imsize, imsize, 3), dtype=np.uint8)
    else:
        path = "masks"
        count = df[path].count()
        train = np.zeros((count, imsize, imsize), dtype=np.uint8)

    for i in tqdm(range(count)):    
        path1 = "human-segmentation-main/" + df[path][i]
        img = Image.open(path1).resize((imsize, imsize), Image.LANCZOS)
        train[i] = np.array(img)
    return train

## Explanation
Dynamically loads either training images or masks based on the data argument.
Each image is resized to 128x128 and stored in a numpy array.
Uses a progress bar to indicate loading progress.

# 6. Building the U-Net Model
Code
Python
inputs = tf.keras.Input((width, height, channels))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

# Encoder
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
# ... (more layers of the U-Net architecture)
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

## Explanation
Implements the U-Net architecture for image segmentation:
Encoder: Extracts features using convolutional layers.
Decoder: Upsamples the features back to the original image size to predict masks.
Uses binary cross-entropy as the loss function and Adam optimizer for training.

# 7. Training the Model Code
Python
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=25, callbacks=callbacks)

## Explanation
Trains the U-Net model using the preprocessed dataset:
X_train: Training images.
Y_train: Corresponding ground-truth masks.
validation_split: Uses 10% of the training data for validation.
batch_size and epochs: Hyperparameters for training.

# 8. Prediction and Visualization
Code
Python
img = Image.open(r"E:\jupyter\projects\Face detection\data\train\images\img23.png")
img = img.resize((128, 128), Image.LANCZOS)
img = np.expand_dims(np.array(img), axis=0)
pre = model1.predict(img, verbose=1)
plt.imshow(pre[0])

## Explanation
Loads a test image, resizes it, and predicts its segmentation mask using the trained model.
Visualizes the predicted mask.

# 9. Overlapping Foreground and Background Code
Python
def overlap(img, mask):
    overlapped = []
    shape = img.shape
    for i in range(shape[0]):
        l1 = []
        for j in range(shape[1]):
            tem = img[i][j] * int(mask[i][j])
            l1.append(tem)
        overlapped.append(l1)
    return np.array(overlapped)

def overlap_with_background(img):
    bac = plt.imread(r"D:\Desktop\OIP.jpg")
    bac = Image.fromarray(bac).resize((128, 128), Image.LANCZOS)
    bac = np.array(bac)
    overlapped = []
    for i in range(img.shape[0]):
        l1 = []
        for j in range(img.shape[1]):
            if np.all(img[i][j] == 0):
                tem = bac[i][j]
            else:
                tem = img[i][j]
            l1.append(tem)
        overlapped.append(l1)
    return np.array(overlapped)

## Explanation
overlap: Combines the image and mask by retaining pixels in the foreground and masking other regions.
overlap_with_background: Replaces the background with a specified image.

# 10. Live Video Background Replacement
Code
Python
import cv2
cap = cv2.VideoCapture(0)
while True:
    score, frame = cap.read()
    frame = cv2.flip(frame, 1)
    # Process and predict the mask
    # Apply background replacement
    if cv2.waitKey(1) == ord("q"):
        break
    cv2.imshow("CAM1", frame)
cap.release()
cv2.destroyAllWindows()

## Explanation
Captures live video from the webcam using OpenCV.
Processes each frame, predicts a segmentation mask, and replaces the background in real-time.
Pressing the q key exits the loop.
