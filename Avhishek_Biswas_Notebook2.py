%matplotlib inline
# We'll start with our library imports...
from __future__ import print_function

import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tqdm import tqdm              # to track progress of loops

class Dense(tf.Module):
    def __init__(self, output_size, activation=tf.nn.relu):
        """
        Args:
            - output_size: (int) number of neurons
            - activation: (function) non-linear function applied to the output
        """
        self.output_size = output_size
        self.activation = activation
        self.is_built = False
        
    def _build(self, x):
        data_size = x.shape[-1]
        self.W = tf.Variable(tf.random.normal([data_size, self.output_size]), name='weights')
        self.b = tf.Variable(tf.random.normal([self.output_size]), name='bias')
        self.is_built = True

    def __call__(self, x):
        if not self.is_built:
            self._build(x)
        return self.activation(tf.matmul(x, self.W) + self.b)

    
DATA_DIR = './tensorflow-datasets/'

ds = tfds.load('mnist', data_dir=DATA_DIR, shuffle_files=True) # this loads a dict with the datasets

# We can create an iterator from each dataset
# This one iterates through the train data, shuffling and minibatching by 32
train_ds = ds['train'].shuffle(1024).batch(32)

# Looping through the iterator, each batch is a dict
for batch in train_ds:
    # The first dimension in the shape is the batch dimension
    # The second and third dimensions are height and width
    # Being greyscale means that the image has one channel, the last dimension in the shape
    print("data shape:", batch['image'].shape)
    print("label:", batch['label'])
    break

# visualize some of the data
idx = np.random.randint(batch['image'].shape[0])
print("An image looks like this:")
imgplot = plt.imshow(batch['image'][idx])
print("It's colored because matplotlib wants to provide more contrast than just greys")

## Splitting the Dataset
# The first 90% of the training data
# Use this data for the training loop
train = tfds.load('mnist', split='train[:90%]', data_dir=DATA_DIR)

# And the last 10%, we'll hold out as the validation set
# Notice the python-style indexing, but in a string and with percentages
# After the training loop, run another loop over this data without the gradient updates to calculate accuracy
validation = tfds.load('mnist', split='train[-10%:]', data_dir=DATA_DIR)

# making the model
first_layer = Dense(200)
second_layer = Dense(10)

loss_values = []
accuracy_values = []
# Loop through one epoch of data
for batch in tqdm(train_ds):
    # run network
    x = tf.reshape(tf.cast(batch['image'], tf.float32), [-1, 784]) # -1 means everyting not otherwise accounted for
    labels = batch['label']
    x = first_layer(x)
    logits = second_layer(x)
    
    # calculate loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss_values.append(loss)
    
    # calculate accuracy
    predictions = tf.argmax(logits, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
    accuracy_values.append(accuracy)

# print accuracy
print("Accuracy:", np.mean(accuracy_values))
# plot per-datum loss
loss_values = np.concatenate(loss_values)
plt.hist(loss_values, density=True, bins=30)
plt.show()


for i in range (2):
    if (i == 0):
        print("lets see how it looks when we run Model 1 with 2 added layers 50 and 100 layers")
        
        # using Sequential groups all the layers to run at once
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(200, tf.nn.relu))
        model.add(tf.keras.layers.Dense(50))
        model.add(tf.keras.layers.Dense(100))
        model.add(tf.keras.layers.Dense(10))
        optimizer = tf.keras.optimizers.Adam()

        loss_values = []
        accuracy_values = []
        # Loop through one epoch of data
        for epoch in range(10):
            for batch in tqdm(train_ds):
                with tf.GradientTape() as tape:
                    # run network
                    x = tf.reshape(tf.cast(batch['image'], tf.float32), [-1, 784])
                    labels = batch['label']
                    logits = model(x)

                    # calculate loss
                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)    
                loss_values.append(loss)

                # gradient update
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                # calculate accuracy
                predictions = tf.argmax(logits, axis=1)
                accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
                accuracy_values.append(accuracy)

        print(model.summary())

        # accuracy
        print("Accuracy:", np.mean(accuracy_values))
        # plot per-datum loss
        loss_values = np.concatenate(loss_values)
        plt.hist(loss_values, density=True, bins=30)
        plt.show()
        
        if (i == 2):
            print("lets see how it looks when we run a Model 2 with different number of neurons")
            
        # using Sequential groups all the layers to run at once
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(10, tf.nn.relu))
        model.add(tf.keras.layers.Dense(10))
        optimizer = tf.keras.optimizers.Adam()

        loss_values = []
        accuracy_values = []
        # Loop through one epoch of data
        for epoch in range(2):
            for batch in tqdm(train_ds):
                with tf.GradientTape() as tape:
                    # run network
                    x = tf.reshape(tf.cast(batch['image'], tf.float32), [-1, 784])
                    labels = batch['label']
                    logits = model(x)

                    # calculate loss
                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)    
                loss_values.append(loss)

                # gradient update
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                # calculate accuracy
                predictions = tf.argmax(logits, axis=1)
                accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
                accuracy_values.append(accuracy)

        print(model.summary())

        # accuracy
        print("Accuracy:", np.mean(accuracy_values))
        # plot per-datum loss
        loss_values = np.concatenate(loss_values)
        plt.hist(loss_values, density=True, bins=30)
        plt.show()
        
        else
        print("lets see how it looks when we run the model with different optimizer here we are using RMSprop")
        
        # using Sequential groups all the layers to run at once
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(200, tf.nn.relu))
        model.add(tf.keras.layers.Dense(10))
        optimizer = tf.keras.optimizers.RMSprop()

        loss_values = []
        accuracy_values = []
        # Loop through one epoch of data
        for epoch in range(1):
            for batch in tqdm(train_ds):
                with tf.GradientTape() as tape:
                    # run network
                    x = tf.reshape(tf.cast(batch['image'], tf.float32), [-1, 784])
                    labels = batch['label']
                    logits = model(x)

                    # calculate loss
                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)    
                loss_values.append(loss)

                # gradient update
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                # calculate accuracy
                predictions = tf.argmax(logits, axis=1)
                accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
                accuracy_values.append(accuracy)

        print(model.summary())

        # accuracy
        print("Accuracy:", np.mean(accuracy_values))
        # plot per-datum loss
        loss_values = np.concatenate(loss_values)
        plt.hist(loss_values, density=True, bins=30)
        plt.show()