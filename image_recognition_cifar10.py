import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import os
import cv2 as cv2
from PIL import Image
from tensorflow import keras
from keras import layers
from sklearn.metrics import ConfusionMatrixDisplay
from collections import defaultdict
from keras.datasets import cifar10

import random
(x_train,y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype("float32")/255.0

y_train = np.squeeze(y_train)

x_test = x_test.astype("float32")/255.0
y_test = np.squeeze(y_test)
height_width = 32
def show_collage(examples):
    box_size = height_width+2
    num_rows,num_cols = examples.shape[:2]
    
    collage = Image.new(
        mode = "RGB",
        size = (num_cols+box_size,num_rows+box_size),
        color = (250,250,250)
    )
    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            array = (np.array(examples[row_idx,col_idx])*255).astype(np.uint8)
            collage.paste(
                Image.fromarray(array), (col_idx * box_size, row_idx * box_size)
            )

    collage = collage.resize((2*num_cols, 2*num_rows*box_size))
    return collage

class_idx_to_train_idxs = defaultdict(list)
for y_train_idx, y in enumerate(y_train):
    class_idx_to_train_idxs[y].append(y_train_idx)
    
    
class_idx_to_test_idxs = defaultdict(list)
for y_test_idx, y in enumerate(y_test):
    class_idx_to_test_idxs[y].append(y_test_idx)
    
num_classes = 10

class AnchorPositivePairs(keras.utils.Sequence):
    def __init__(self, num_batches):
        self.num_batches = num_batches
        
    def __len__(self):
        return self.num_batches
    def __getitem__(self, _idx):
        x = np.empty((2,num_classes,height_width,height_width,3), dtype=np.float32)
        for class_idx in range(num_classes):
            examples_for_class = class_idx_to_train_idxs[class_idx]
            anchor_idx = random.choice(examples_for_class)
            positive_idx = random.choice(examples_for_class)
            while positive_idx == anchor_idx:
                positive_idx = random.choice(examples_for_class)
            x[0, class_idx] = x_train[anchor_idx]
            x[1, class_idx] = x_train[positive_idx] 
        return x

class EmbeddingModel(keras.Model):
    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        anchors, positives = data[0], data[1]
        
        with tf.GradientTape() as tape:
            anchor_embeddings = self(anchors, training= True)
            positive_embeddings = self(positives, training = True)
            
            similarities = tf.einsum("ae,pe->ap",
                anchor_embeddings,positive_embeddings
            )
            
            temparature = 0.2
            similarities/=temparature
            
            sparse_labels = tf.range(num_classes)
            loss = self.compiled_loss(sparse_labels,similarities)
            
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,self.trainable_variables))
        
        self.compiled_metrics.update_state(sparse_labels,similarities)
        
        return {m.name:m.result() for m in self.metrics}
        
        
        
input = layers.Input(shape = (height_width,height_width,3))

x = layers.Conv2D(filters=32,kernel_size=3,strides=2,activation ="relu")(input)
x = layers.Conv2D(filters=64,kernel_size=3,strides=2,activation ="relu")(x)
x = layers.Conv2D(filters=128,kernel_size=3,strides=2,activation ="relu")(x)
x = layers.GlobalAveragePooling2D()(x)
embeddings = layers.Dense(units = 8, activation=None)(x)
embeddings = tf.nn.l2_normalize(embeddings,axis = 1)

model = EmbeddingModel(input, embeddings)


model.compile(
    optimizer=keras.optimizers.Adam(learning_rate = 1e-3),
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True)
    
)

history = model.fit(AnchorPositivePairs(num_batches=1000), epochs = 20)

plt.plot(history.history["loss"])
plt.show()

#TESTING

near_neighbors_per_example = 10
embeddings = model.predict(x_test)
gram_matrix = np.einsum("ae,be->ab", embeddings,embeddings)
near_neighbors = np.argsort(gram_matrix.T)[:,-(near_neighbors_per_example+1):]
 
 
num_collage_examples = 5


examples = np.empty(
    
    (
        num_collage_examples,
        near_neighbors_per_example+1,
        height_width,
        height_width,
        3   
        
    ),
    dtype = np.float32
    
)
for row_idx in range(num_collage_examples):
        
    examples[row_idx,0] = x_test[row_idx]
    anchor_near_neighbours = reversed(near_neighbors[row_idx][:-1])
    for col_idx, nn_idx in enumerate(anchor_near_neighbours):
        examples[row_idx, col_idx+1] = x_test[nn_idx]
        
show_collage(examples)


confusion_matrix = np.zeros((num_classes, num_classes))

# For each class.
for class_idx in range(num_classes):
    # Consider 10 examples.
    example_idxs = class_idx_to_test_idxs[class_idx][:10]
    for y_test_idx in example_idxs:
        # And count the classes of its near neighbours.
        for nn_idx in near_neighbors[y_test_idx][:-1]:
            nn_class_idx = y_test[nn_idx]
            confusion_matrix[class_idx, nn_class_idx] += 1

# Display a confusion matrix.
labels = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
]
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)
disp.plot(include_values=True, cmap="viridis", ax=None, xticks_rotation="vertical")
plt.show()

