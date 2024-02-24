import os
import json
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

  
  
train_path = 'dataset/train_data'
test_path = 'dataset/test_data'


import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data

from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.transform import rescale


def change(path):
  original = plt.imread(path)
  original_scaled = rescale(original, 0.125, anti_aliasing=True, channel_axis=2)
  image = rgb2gray(original_scaled)
  image = img_as_ubyte(image)


  edges = canny(image, sigma=3)


  # Detect two radii
  hough_radii = np.arange(180, 220, 2)
  hough_res = hough_circle(edges, hough_radii)

  # Select the most prominent 3 circles
  accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                            total_num_peaks=1)

  # Draw them
  fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
  image = color.gray2rgb(image)
  for center_y, center_x, radius in zip(cy, cx, radii):
      circy, circx = circle_perimeter(center_y, center_x, radius,
                                      shape=image.shape)
      image[circy, circx] = (220, 20, 20)

  ax.imshow(image, cmap=plt.cm.gray)
  plt.show()

  # Create an empty mask with the same dimensions as the image
  mask = np.zeros_like(image)

  # Use the 'circle' function from 'skimage.draw' to get the coordinates of the filled circle
  rr, cc = draw.disk((center_y, center_x), radius, shape=image.shape)

  # Set the mask to 1 (or 255 for a binary image) inside the circle
  mask[rr, cc] = 255
  

  original_scaled[mask==0]=0

  return original_scaled

# Load data
def load_train_data(path,train_data_num=2000):
    images = []
    labels = []
    files=os.listdir(path)
    files = list(set([i.split(".")[0] for i in files if i.split(".")[0]]))
    i=0
    
    for img_file in files:
        if i % 1000 == 0:
            print(f'{i} done')
        if i==train_data_num and train_data_num!=0:
            break
        
        #print(img_file)
        
        # Save label from json
        json_file = path + "/" + img_file + '.json'
        
    
        with open(json_file) as f:
            data = json.load(f)
            colonies_count = data['colonies_number']
        labels.append(0 if colonies_count == 0 else 1)
        
        # Load and preprocess the image
        image_file = path + "/" + img_file + '.jpg'
        img = Image.open(image_file)
        img = img.resize((128, 128)) # Resize image to 128x128
        img = np.array(img) / 255.0 # Normalize pixel values
        
        images.append(img)
        
        i+=1
        
    return np.array(images), np.array(labels)

def load_test_data(path,test_data_num=0):
    images = []
    IDs = []
    files=os.listdir(path)
    #files = list(set([i.split(".")[0] for i in files if i.split(".")[0]]))
    i=0
    
    for img_file in files:
        if i==test_data_num and test_data_num!=0:
            break
        
        # Load and preprocess the image
        image_file = path + "/" + img_file
        img = Image.open(image_file)
        img = img.resize((128, 128)) # Resize image to 128x128
        img = np.array(img) / 255.0 # Normalize pixel values
        
        images.append(img)
        IDs.append(img_file.split(".")[0])
        
        i+=1
        
    return np.array(images),IDs

print("start loading")
images, labels = load_train_data(train_path)

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


print("start training")

# Train model function
def train(images,labels):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc*100:.2f}%")

images,labels = load_train_data(train_path)
train(images,labels)


test_images,IDs = load_test_data(test_path)
predictions = model.predict(test_images)
predicted_labels = (predictions > 0.5).astype(int)
predicted_labels = [i[0] for i in predicted_labels]


data = {
    'ID': IDs,
    'TARGET': predicted_labels
}

df = pd.DataFrame(data)
df.to_csv("output.csv", sep=',', index=False, encoding='utf-8')



