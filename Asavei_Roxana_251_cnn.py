#importing necessary libraries
import numpy as np
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



def read_images(folder, target_size=(224, 224)): #reading images
    df = pd.read_csv(folder + '.csv')
    file_names = df['image_id'].astype(str).to_list()
    images = []
    for file_name in file_names:
        img_path = os.path.join(folder, file_name + '.png')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # converting them to RGB format
        img = cv2.resize(img, target_size) #resizing them to 224 x 224
        images.append(img)
    return np.array(images)

def read_labels(file_name):
    df = pd.read_csv(file_name + '.csv')
    return df['label'].astype(int).to_numpy()

def plotting_accuracy(history):
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(train_acc) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, 'b-o', label='Acuratețe antrenare')
    plt.plot(epochs, val_acc, 'g-o', label='Acuratețe validare')

    plt.title('Evoluția acurateții pe epoci')
    plt.xlabel('Epocă')
    plt.ylabel('Acuratețe')
    plt.xticks(epochs)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def conf_matrix(model, validation_images, validation_labels):
    validation_pred = model.predict(validation_images)
    valid_pred_classes = np.argmax(validation_pred, axis=1)
    cm = confusion_matrix(validation_labels, valid_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', values_format='d') 
    plt.title("Matricea de confuzie")
    plt.xlabel("Etichete prezise")
    plt.ylabel("Etichete reale")
    plt.tight_layout()
    plt.show()

def predict_test(model, test_images):
    pred_probs = model.predict(test_images) #predicting results         
    pred_labels = np.argmax(pred_probs, axis=1)  #converting them to labels

    df_test = pd.read_csv('test.csv')              
    df_test['label'] = pred_labels                       

    df_test[['image_id', 'label']].to_csv('predictionsCNN.csv', index=False)

#uploading and normalizing images
train_images = read_images(folder='train')
validation_images = read_images(folder='validation')
test_images = read_images(folder="test")
train_images = train_images.astype('float32') / 255.0
validation_images = validation_images.astype('float32') / 255.0
test_images = validation_images.astype('float32') / 255.0

#reading and processing labels
train_labels = read_labels('train')
validation_labels = read_labels('validation')
num_classes = len(np.unique(train_labels))
train_labels_cat = to_categorical(train_labels, num_classes)
validation_labels_cat = to_categorical(validation_labels, num_classes)

#creating model
model7 = Sequential()
model7.add(Conv2D(32, (3,3), 1, activation='relu', input_shape=(224,224,3))) #applying 32 filters
model7.add(MaxPooling2D((2, 2))) #reducing size
model7.add(BatchNormalization()) #mornalization

model7.add(Conv2D(64, (3,3), 1, activation='relu'))
model7.add(MaxPooling2D((2, 2)))
model7.add(BatchNormalization())

model7.add(Conv2D(128, (3,3), 1, activation='relu'))
model7.add(MaxPooling2D((2, 2)))
model7.add(BatchNormalization())

model7.add(Conv2D(256, (3,3), 1, activation='relu'))
model7.add(AveragePooling2D(2, 2))
model7.add(BatchNormalization())

model7.add(GlobalAveragePooling2D()) #average pool on each feature map

model7.add(Flatten())
model7.add(Dense(512, activation='relu'))
model7.add(Dropout(0.5)) #deactivating 50% of neutrons to reduce overfitting
model7.add(Dense(num_classes, activation='softmax'))

#checking model structure
model7.summary()

#compiling model
model7.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

#defining callbacks
#reducing learning rate each epoch val_loss doesnt improve
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,      
    patience=1,     
    min_lr=1e-9     
)
#waiting 5 epochs for val_accuracy to improve, else we stop
early_stop = EarlyStopping(
    monitor='val_accuracy',       
    patience=5,               
    restore_best_weights=True 
)

#training model
history = model7.fit(
    train_images, train_labels_cat,
    epochs=30,
    batch_size=32,
    validation_data=(validation_images, validation_labels_cat),
    callbacks=[reduce_lr, early_stop]
)

plotting_accuracy(history)
#this confusion matrix is made on the model only after 30 epochs
conf_matrix(model7, validation_images, validation_labels) 
predict_test(model7, train_images)
