import tensorflow as tf
import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tqdm import tqdm              # to track progress of loops

EXPANSION_FACTOR = 4

class Bottleneck(tf.Module):
    
    def __init__(self, filter_num, stride=1):
        super(Bottleneck, self).__init__()
        self.stride = stride
        assert filter_num % EXPANSION_FACTOR == 0
        
        self.conv1 = tf.keras.layers.Conv2D(filter_num // EXPANSION_FACTOR, kernel_size=1, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filter_num // EXPANSION_FACTOR, kernel_size=3, strides=stride, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filter_num, kernel_size=1, padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()
        
        if self.stride != 1:
            self.down_conv = tf.keras.layers.Conv2D(filters=filter_num,
                                                    kernel_size=(1, 1),
                                                    strides=stride,
                                                    padding="same")
            self.down_bn = tf.keras.layers.BatchNormalization()

    def __call__(self, x, is_training):
        identity = x
        if self.stride != 1:
            identity = self.down_conv(identity)
            identity = self.down_bn(identity, training=is_training)

        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)

        x = self.bn3(x)
        x = tf.nn.relu(x)
        x = self.conv3(x)

        return x + identity

class MyModel(tf.keras.Model):
    def __init__(self, num_classes):
        super(MyModel, self).__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')
        
        self.block1a = Bottleneck(64, stride=1)
        self.block1b = Bottleneck(64, stride=1)
        
        self.block2a = Bottleneck(128, stride=2)
        self.block2b = Bottleneck(128, stride=1)
        
        self.block3a = Bottleneck(256, stride=2)
        self.block3b = Bottleneck(256, stride=1)
        self.block3c = Bottleneck(256, stride=1)
        
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')
        
    def call(self, x, training=True):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        
        x = self.block1a(x, training=training)
        x = self.block1b(x, training=training)
        
        x = self.block2a(x, training=training)
        x = self.block2b(x, training=training)
        
        x = self.block3a(x, training=training)
        x = self.block3b(x, training=training)
        x = self.block3c(x, training=training)
        
        x = self.avg_pool(x)
        x = self.fc(x)
        
        return x
    
    
    
model = MyModel(num_classes=10)