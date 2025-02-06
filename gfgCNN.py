# import the necessary libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from itertools import product

# set the param 
plt.rc('figure', autolayout=True)
plt.rc('image', cmap='magma')

# Define multiple filters instead of a single kernel
kernels = tf.constant([
    [[[-1.0], [-1.0], [-1.0]],  # Edge detection
     [[-1.0], [8.0], [-1.0]],
     [[-1.0], [-1.0], [-1.0]]],

    [[[0.0], [-1.0], [0.0]],  # Sobel filter (vertical edges)
     [[-1.0], [4.0], [-1.0]],
     [[0.0], [-1.0], [0.0]]],

    [[[-1.0], [0.0], [1.0]],  # Sobel filter (horizontal edges)
     [[-2.0], [0.0], [2.0]],
     [[-1.0], [0.0], [1.0]]]
])

# Convert to TensorFlow format
kernels = tf.reshape(kernels, [3, 3, 1, 3])  # Three filters
kernels = tf.cast(kernels, dtype=tf.float32)

# load the image
image = tf.io.read_file('dog.jpg')
image = tf.io.decode_jpeg(image, channels=1)
image = tf.image.resize(image, size=[300, 300])

# plot the image
img = tf.squeeze(image).numpy()
plt.figure(figsize=(5, 5))
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title('Original Gray Scale image')
plt.show()

# Reformat
image = tf.image.convert_image_dtype(image, dtype=tf.float32)
image = tf.expand_dims(image, axis=0)

# Apply convolution with multiple filters
image_filter = tf.nn.conv2d(image, filters=kernels, strides=1, padding='SAME')

plt.figure(figsize=(15, 5))

# Plot the convolved images for each filter
for i in range(kernels.shape[0]):
    plt.subplot(1, 3, i + 1)
    plt.imshow(tf.squeeze(image_filter[0, :, :, i]))
    plt.axis('off')
    plt.title(f'Convolution with Filter {i + 1}')

plt.show()

# activation layer
relu_fn = tf.nn.relu
# Image detection
image_detect = relu_fn(image_filter)

plt.figure(figsize=(15, 5))

# Plot the activated images
for i in range(kernels.shape[0]):
    plt.subplot(1, 3, i + 1)
    plt.imshow(tf.squeeze(image_detect[0, :, :, i]))
    plt.axis('off')
    plt.title(f'Activation with Filter {i + 1}')

plt.show()

# Pooling layer
pool = tf.nn.pool
image_condense = pool(input=image_detect, 
                      window_shape=(2, 2),
                      pooling_type='MAX',
                      strides=(2, 2),
                      padding='SAME')

plt.figure(figsize=(15, 5))

# Plot the pooled images
for i in range(kernels.shape[0]):
    plt.subplot(1, 3, i + 1)
    plt.imshow(tf.squeeze(image_condense[0, :, :, i]))
    plt.axis('off')
    plt.title(f'Pooling with Filter {i + 1}')

plt.show()