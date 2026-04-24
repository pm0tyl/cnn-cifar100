from keras import datasets
from keras.datasets import cifar10
from keras.datasets import mnist
from keras import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input,Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()

tf.debugging.set_log_device_placement(False)

class_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver',
'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge',
'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab',
'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish',
'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man',
'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom',
'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper',
'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale',
'willow_tree', 'wolf', 'woman', 'worm']

x_train=train_images = train_images.astype('float32') / 255
x_test=test_images = test_images.astype('float32') / 255
y_train=train_labels = to_categorical(train_labels)
y_test=test_labels = to_categorical(test_labels)

model = Sequential([
    Input(shape=(32,32,3)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(256, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(256, activation='relu'),   
    Dropout(0.5),     
    Dense(128, activation='relu'),  
    Dropout(0.5),              
    Dense(100, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0005),
loss='categorical_crossentropy',
metrics=['accuracy'])

early_stop = EarlyStopping(
    monitor='val_accuracy',   
    patience=5,               
    restore_best_weights=True 
)

history = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test),validation_split=0.1,batch_size=64, verbose=1,callbacks=[early_stop])

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val_train')
plt.title('Loss'); plt.xlabel('epoch'); plt.legend(); plt.show()
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val_train')
plt.title('Accuracy'); plt.xlabel('epoch'); plt.legend(); plt.show()

predict_x = model.predict(x_train)
y_result = np.argmax(predict_x,axis=1)

y_train = np.argmax(y_train, axis=1)
accuracy = accuracy_score(y_train, y_result)
precision = precision_score(y_train, y_result, average='weighted')
recall = recall_score(y_train, y_result, average='weighted')
conf_matrix = confusion_matrix(y_train, y_result)

print("\nOcena na zbiorze treningowym:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("Confusion Matrix:\n", conf_matrix)


predict_x2 = model.predict(x_test)
y_result2 = np.argmax(predict_x2,axis=1)

y_test = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_test, y_result2)
precision = precision_score(y_test, y_result2, average='weighted')
recall = recall_score(y_test, y_result2, average='weighted')
conf_matrix = confusion_matrix(y_test, y_result2)

print("\nOcena na zbiorze testowym:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("Confusion Matrix:\n", conf_matrix)