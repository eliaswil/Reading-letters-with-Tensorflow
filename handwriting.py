# Following the tensorflow tutorial:
# https://www.tensorflow.org/tutorials/keras/classification
#
# Dataset from: https://www.nist.gov/itl/products-and-services/emnist-dataset
# -> dataset description: http://yann.lecun.com/exdb/mnist/


import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import cv2

# import os

path_train_labels = 'gzip/emnist-letters-train-labels-idx1-ubyte'
path_train_images = 'gzip/emnist-letters-train-images-idx3-ubyte'

path_test_labels = 'gzip/emnist-letters-test-labels-idx1-ubyte'
path_test_images = 'gzip/emnist-letters-test-images-idx3-ubyte'

header_labels = 8
header_images = 16
offset = 0

class_names = [chr(i+65) for i in range(26)]



def load_labels(path, no_labels, offset, header=8, show=False):
    f_train_labels = open(path, 'rb')
    f_train_labels.read(header + offset)
    
    labels = []
    
    for i in range(no_labels):
        labels.append(ord(f_train_labels.read(1))-1)
        if show:
            print(f'[{i}]: {labels[-1]}')
    return np.array(labels)


def load_images(path, no_imgs, offset, header=16, show=False):
    f_train_images = open(path, 'rb')
    f_train_images.read(header + 28*28*offset)
    
    images = []

    for img_no in range(no_imgs):
        image = []

        for i in range(28):
            row = []
            for j in range(28):
                row.append(ord(f_train_images.read(1)))
            image.append(row)
            
        # print(image)
        image = np.array(image)
        image = image.astype(np.uint8)
        image = np.transpose(image)
        
        images.append(image)

        if show:
            cv2.imshow('img', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    return np.array(images)
        
# load data
print("Loading data...")
train_labels = load_labels(path_train_labels, no_labels=60000, offset=0, header=8, show=False)
train_images = load_images(path_train_images, no_imgs=60000, offset=0, header=16, show=False)

test_labels = load_labels(path_test_labels, no_labels=10000, offset=0, header=8, show=False)
test_images = load_images(path_test_images, no_imgs=10000, offset=0, header=16, show=False)
print("Data loaded.")

# scale to [0;1]
train_images = train_images / 255.0
test_images = test_images / 255.0

def display_images(images, labels):
    plt.figure(figsize=(10,10))
    for i in range(len(images)):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])
    plt.show()
    
# display_images(train_images[:25], train_labels[:25])




#### Build the model


# # set up the layers
# print('Building the model...')
# model = tf.keras.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)), # from 2d-array -> 1d-array
#     tf.keras.layers.Dense(128, activation='relu'), # densely (fully) connected neural layers (128 neurons)
#     tf.keras.layers.Dense(26)
# ])


# # compile the model
# print('Compiling the model...')
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
# print('Model compiled.')

# #### Train the model

# print('Training the model...')
# model.fit(train_images, train_labels, epochs=10)
# print('Model trained.')


### Load a model
print('Loading a model...')
model = tf.keras.models.load_model('model_handwriting.h5')


### Visualize

def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(26))
    plt.yticks([])
    thisplot = plt.bar(range(26), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
    
def verify_visually(offset, rows, cols):     
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(offset, predictions[offset], test_labels, test_images)
    plt.subplot(1,2,2)
    plot_value_array(offset, predictions[offset],  test_labels)
    plt.show()
    
    # Plot the first X test images, their predicted labels, and the true labels.
    # Color correct predictions in blue and incorrect predictions in red.
    num_rows = rows
    num_cols = cols
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i+offset, predictions[i+offset], test_labels, test_images)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i+offset, predictions[i+offset], test_labels)
    plt.tight_layout()
    plt.show()

### Evaluate accuracy

print('Evaluating the model...')
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy: ', test_acc)

# make predictions
print('Making some predictions...')
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(predictions[0])
print(np.argmax(predictions[0]))
print(test_labels[0])

# make a single prediction
print('A Single prediction..')
img = test_images[0] # any image of size (28x28)
img = (np.expand_dims(img, 0))
predictions_single = probability_model.predict(img)
print(predictions_single)
np.argmax(predictions_single[0])
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plt.imshow(img[0])
plt.subplot(1,2,2)
plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(26), class_names, rotation=45)
plt.show()






    
verify_visually(5000, 5, 1)

# save the model
print('Saving the model...')
model.save('model_handwriting.h5')
print('Model saved.')