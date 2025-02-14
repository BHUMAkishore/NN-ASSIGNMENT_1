#!/usr/bin/env python
# coding: utf-8

# In[2]:


#QUESTION_1

import tensorflow as tf

# 1. Create a random tensor of shape (4, 6)
tensor = tf.random.uniform(shape=(4, 6))

# 2. Find its rank and shape using TensorFlow functions
rank = tf.rank(tensor)
shape = tf.shape(tensor)
print(f"Original Tensor:\n{tensor}\n")
print(f"Rank: {rank.numpy()}, Shape: {shape.numpy()}\n")

# 3. Reshape it into (2, 3, 4) and transpose it to (3, 2, 4)
reshaped_tensor = tf.reshape(tensor, (2, 3, 4))
transposed_tensor = tf.transpose(reshaped_tensor, perm=[1, 0, 2])
print(f"Reshaped Tensor:\n{reshaped_tensor}\n")
print(f"Transposed Tensor:\n{transposed_tensor}\n")

# 4. Broadcast a smaller tensor (1, 4) to match the larger tensor and add them
small_tensor = tf.random.uniform(shape=(1, 4))
broadcasted_tensor = tf.broadcast_to(small_tensor, (3, 2, 4))
added_tensor = transposed_tensor + broadcasted_tensor
print(f"Broadcasted Tensor:\n{broadcasted_tensor}\n")
print(f"Result after Addition:\n{added_tensor}\n")

# 5. Explanation of Broadcasting
explanation = """
Broadcasting in TensorFlow allows operations between tensors of different shapes
by automatically expanding the smaller tensor to match the larger tensor's shape.
In this case, the (1, 4) tensor was expanded to (3, 2, 4) to align with the
transposed tensor's shape before performing element-wise addition.
"""
print(explanation)


# In[3]:


#QUESTION_2

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. Define true values (y_true) and model predictions (y_pred)
y_true = tf.constant([0, 1, 1, 0], dtype=tf.float32)
y_pred1 = tf.constant([0.1, 0.9, 0.8, 0.2], dtype=tf.float32)
y_pred2 = tf.constant([0.2, 0.8, 0.7, 0.3], dtype=tf.float32)

# 2. Compute Mean Squared Error (MSE) and Categorical Cross-Entropy (CCE) losses
mse1 = tf.keras.losses.MeanSquaredError()(y_true, y_pred1)
cce1 = tf.keras.losses.CategoricalCrossentropy()(tf.one_hot(tf.cast(y_true, tf.int32), depth=2), tf.one_hot(tf.cast(y_pred1 > 0.5, tf.int32), depth=2))

mse2 = tf.keras.losses.MeanSquaredError()(y_true, y_pred2)
cce2 = tf.keras.losses.CategoricalCrossentropy()(tf.one_hot(tf.cast(y_true, tf.int32), depth=2), tf.one_hot(tf.cast(y_pred2 > 0.5, tf.int32), depth=2))

print(f"MSE for first prediction: {mse1.numpy()}")
print(f"CCE for first prediction: {cce1.numpy()}")
print(f"MSE for second prediction: {mse2.numpy()}")
print(f"CCE for second prediction: {cce2.numpy()}")

# 3. Plot loss function values using Matplotlib
labels = ['MSE (Pred1)', 'CCE (Pred1)', 'MSE (Pred2)', 'CCE (Pred2)']
values = [mse1.numpy(), cce1.numpy(), mse2.numpy(), cce2.numpy()]

plt.figure(figsize=(8, 5))
plt.bar(labels, values, color=['blue', 'orange', 'blue', 'orange'])
plt.xlabel('Loss Type')
plt.ylabel('Loss Value')
plt.title('Comparison of MSE and CCE Loss')
plt.show()


# In[4]:


#QUESTION_3
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define a simple neural network model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Train two models: One with Adam and another with SGD
optimizers = {'Adam': tf.keras.optimizers.Adam(), 'SGD': tf.keras.optimizers.SGD()}
history = {}

for opt_name, optimizer in optimizers.items():
    model = create_model()
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print(f"Training with {opt_name}...")
    history[opt_name] = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), verbose=2)

# Compare training and validation accuracy trends
plt.figure(figsize=(10, 5))
for opt_name in optimizers.keys():
    plt.plot(history[opt_name].history['accuracy'], label=f'{opt_name} Train')
    plt.plot(history[opt_name].history['val_accuracy'], label=f'{opt_name} Val')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Comparison of Adam vs. SGD Performance')
plt.legend()
plt.show()


# In[ ]:



#QUESTION_4


# In[6]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define a simple neural network model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Set up TensorBoard logging
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

# Train two models: One with Adam and another with SGD
optimizers = {'Adam': tf.keras.optimizers.Adam(), 'SGD': tf.keras.optimizers.SGD()}
history = {}

for opt_name, optimizer in optimizers.items():
    model = create_model()
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print(f"Training with {opt_name}...")
    history[opt_name] = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), 
                                  verbose=2, callbacks=[tensorboard_callback])

# Compare training and validation accuracy and loss trends
plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
for opt_name in optimizers.keys():
    plt.plot(history[opt_name].history['accuracy'], label=f'{opt_name} Train')
    plt.plot(history[opt_name].history['val_accuracy'], label=f'{opt_name} Val')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison: Adam vs. SGD')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
for opt_name in optimizers.keys():
    plt.plot(history[opt_name].history['loss'], label=f'{opt_name} Train')
    plt.plot(history[opt_name].history['val_loss'], label=f'{opt_name} Val')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Comparison: Adam vs. SGD')
plt.legend()

plt.show()

print("Run the following command in a terminal to launch TensorBoard:")
print("tensorboard --logdir=logs/fit")


# In[ ]:




