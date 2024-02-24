import os
import json
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

train_path = 'dataset/train_data'
# Load data
def load_data(path):
    images = []
    labels = []
    files=os.listdir(path)
    files = list(set([i.split(".")[0] for i in files if i.split(".")[0]]))
    i=0
    train_data_num=4000
    
    for img_file in files:
        if i % 1000 == 0:
            print(f'{i} done')
        if i==train_data_num and train_data_num!=0:
            break
        
        #(img_file)
        
        
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

print("start loading")
images, labels = load_data(train_path)

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
# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc*100:.2f}%")
