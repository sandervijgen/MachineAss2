import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Define the path to your dataset folders
train_folder = 'train/'
test_folder = 'test/'

# Function to load images and labels from a folder
def load_images_and_labels(folder):
    images = []
    labels = []
    class_names = os.listdir(folder)
    for class_name in class_names:
        class_path = os.path.join(folder, class_name)
        for filename in os.listdir(class_path):
            if filename.endswith('.png'):
                image_path = os.path.join(class_path, filename)
                image = Image.open(image_path)
                image = image.resize((32, 32))  # Resize if needed
                image_array = np.array(image)
                images.append(image_array)
                labels.append(class_name)
    return np.array(images), np.array(labels)

# Load training data
IMA_train, LAB_train = load_images_and_labels(train_folder)

# Load test data
IMA_test, LAB_test = load_images_and_labels(test_folder)

# Convert labels to categorical
LAB_train = to_categorical(LAB_train)
LAB_test = to_categorical(LAB_test)

# Normalize pixel values to between 0 and 1
IMA_train = IMA_train / 255.0
IMA_test = IMA_test / 255.0

# Display some examples
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(IMA_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[np.argmax(LAB_train[i])])
plt.show()

# Continue with the rest of your code for model creation, compilation, training, and evaluation
# ...

# Remember to adjust the input shape in your model to (32, 32, 3) if it's not already
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))

