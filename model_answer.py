from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class CNN_Model(object):
    def __init__(self, weight_path=None):
        self.weight_path = weight_path if weight_path is not None else 'weight.keras'  # Default path
        self.model = None

    def build_model(self, rt=False):
        self.model = Sequential()
        self.model.add(Input(shape=(28, 28, 1)))
        self.model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.1))
        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.3))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2, activation='softmax'))
        
        if self.weight_path is not None and Path(self.weight_path).exists():
            self.model.load_weights(self.weight_path)
        
        if rt:
            return self.model

    @staticmethod
    def load_data():
        dataset_dir = './create_dataset/dataset/'
        images = []
        labels = []
        for img_path in Path(dataset_dir + 'unchoice/').glob("*.png"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28), cv2.INTER_AREA)
            img = img.reshape((28, 28, 1))
            label = to_categorical(0, num_classes=2)
            images.append(img / 255.0)
            labels.append(label)
        for img_path in Path(dataset_dir + 'choice/').glob("*.png"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28), cv2.INTER_AREA)
            img = img.reshape((28, 28, 1))
            label = to_categorical(1, num_classes=2)
            images.append(img / 255.0)
            labels.append(label)
        datasets = list(zip(images, labels))
        np.random.shuffle(datasets)
        images, labels = zip(*datasets)
        return np.array(images), np.array(labels)

    def train(self):
        images, labels = self.load_data()
        self.build_model()
        self.model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(1e-3), metrics=['accuracy'])
        reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=5, verbose=1)
        cpt_save = ModelCheckpoint(self.weight_path, save_best_only=True, monitor='val_accuracy', mode='max')
        print("Training......")
        self.model.fit(images, labels, callbacks=[cpt_save, reduce_lr], verbose=1, epochs=20, 
                      validation_split=0.2, batch_size=32, shuffle=True)
        
        

if __name__ == "__main__":
    model = CNN_Model()  # Uses default 'weight.keras'
    model.train()