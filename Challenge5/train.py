import numpy as np
import os
import PIL
import tensorflow as tf
import shutil

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import pandas as pd


batch_size = 300
img_height = 32
img_width = 32

data_dir=pathlib.Path('../data/challenge5/train_images_sorted/')

# prep training sample
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# prep validation sample
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

train_ds = train_ds.cache() # met dataset dans cache
train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE) # prefetch du cpu vers gpu pendant training

# augment data
data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical",
                                                 input_shape=(img_height,
                                                              img_width,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
    layers.experimental.preprocessing.RandomContrast(0.3),

  ]
)

num_classes=10

# try another model with dropout
model = Sequential([
    data_augmentation,
    layers.experimental.preprocessing.Rescaling(1./255),
    tf.keras.layers.BatchNormalization(),
    layers.Conv2D(64, 3, padding='same', activation=tf.keras.layers.LeakyReLU()),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation=tf.keras.layers.LeakyReLU()),
    layers.Conv2D(64, 3, padding='same', activation=tf.keras.layers.LeakyReLU()),
    tf.keras.layers.BatchNormalization(),
    layers.Conv2D(128, 3, padding='same', activation=tf.keras.layers.LeakyReLU()),
    layers.Conv2D(128, 3, padding='same', activation=tf.keras.layers.LeakyReLU()),
    tf.keras.layers.BatchNormalization(),
    layers.Conv2D(64, 3, padding='same', activation=tf.keras.layers.LeakyReLU()),
    layers.Conv2D(64, 3, padding='same', activation=tf.keras.layers.LeakyReLU()),
    tf.keras.layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='./checkpoint/',
    save_weights_only=True,
    save_freq='epoch')
#    monitor='val_acc',
#    mode='max',
#    save_best_only=True)


epochs = 100
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks = [model_checkpoint_callback]
)
test_data_dir=pathlib.Path('../data/challenge5/test_images/')
test_image_count = len(list(test_data_dir.glob('*.png')))
print(test_image_count)

onlyfiles = [f for f in os.listdir(test_data_dir) if os.path.isfile(os.path.join(test_data_dir, f))] # files to sort
onlyfiles.sort(key=key)

test_data_list = []

for i in range(test_image_count):
    filepath = '../data/challenge5/test_images/' + onlyfiles[i]

    img = keras.preprocessing.image.load_img(filepath, target_size=(img_height, img_width))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    test_data_list.append(img_array)

test_data = tf.data.Dataset.from_tensor_slices(test_data_list)

score = model.predict(test_data)


with open('predictions.txt','w') as newfile:
    newfile.write('id,classes\n')
    for i in range(test_image_count):
        newfile.write('{},{}\n'.format(i,np.argmax(score[i])))
