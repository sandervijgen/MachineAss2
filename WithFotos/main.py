import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.src.utils import to_categorical
from tensorflow.keras import layers, models, datasets
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#load dataset image and labels train en test
(IMA_train,LAB_train), (IMA_test,LAB_test) = datasets.cifar10.load_data()

#pixel value between 0 and 255 to 0 and 1
IMA_train, IMA_test = IMA_train / 255.0, IMA_test / 255.0

#vector to matrix
LAB_train = to_categorical(LAB_train, 10)
LAB_test = to_categorical(LAB_test, 10)

class_names = ['plane', 'car', 'bird' , 'cat', 'deer', 'dog', 'frog', 'horse', 'ship','truck']

#pick first 40000 images of train and 8000 of test so it is smaller and faster
IMA_train = IMA_train[:40000]
LAB_train = LAB_train[:40000]
IMA_test = IMA_test[:8000]
LAB_test = LAB_test[:8000]

#show some train pictures as example
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(IMA_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[np.argmax(LAB_train[i])])
plt.show()

# Make the model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
FotoModel = model.fit(IMA_train, LAB_train, epochs=10, validation_data=(IMA_test, LAB_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(IMA_test, LAB_test, verbose=2)
print(f"\nTest accuracy: {test_acc}")

