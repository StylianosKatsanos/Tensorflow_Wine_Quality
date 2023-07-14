# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 18:58:45 2023

@author: Stelios

@ONLINE {cortezpaulo;cerdeiraantonio;almeidafernando;matostelmo;reisjose1999,
    author = "Cortez, Paulo; Cerdeira, Antonio; Almeida,Fernando;  Matos, Telmo;  Reis, Jose",
    title  = "Modeling wine preferences by data mining from physicochemical properties.",
    year   = "2009",
    url    = "https://archive.ics.uci.edu/ml/datasets/wine+quality"
}
"""

#------------------------  Module Importing --------------------------------------------#

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#------------------------  Accesing Data ---------------------------#

dataset = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', delimiter = ";")

#-----------------------  Preprocessing Data -----------------------------------------#

features_train = dataset.sample(frac=0.8,random_state=0) # takes a sample of the database of 80 % for trainning

features_test = dataset.drop(features_train.index) # drops the trainning sample to keep the 20 % for testing

#Quality is the output of the model so it is stored on its own dataset

labels_train = features_train.pop('quality')

labels_test = features_test.pop('quality')

#--------------------------  Building Model ---------------------------------------#

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(features_train))

def build_model():
    model = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(64, activation=tf.nn.relu, input_shape=[len(features_train.keys()),]),  #input shape the rows of the datasets
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(1)
    ])
    
    
    optimizer = tf.keras.optimizers.Adam(0.001)
        
    model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mae', 'mse'])
    return model


model = build_model()
#------------------------- Fitting Data into Model ---------------------------------#

EPOCHS = 100

history = model.fit(
    features_train, labels_train,
    epochs=EPOCHS, validation_split=0.2, verbose=0)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

#------------------------------ Plotting Data ----------------------------------#

def plot_history(history):
      
    #Plot 1 -- Mean Absoloute Error - Epoch
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [Quality]')
    plt.plot(hist['epoch'], hist['mae'],
                 label = 'Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
                 label = 'val Error')
    plt.legend()
    plt.ylim([0,5])
      
    
    #Plot 2 -- Mean Squared Error - Epoch
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$Quality^2$]')
    plt.plot(hist['epoch'], hist['mse'],
                 label='train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
                 label = 'Val Error')
    plt.legend()
    plt.ylim([0,5])

plot_history(history) 

#-------------------------  Evaluating Model with Test Data ----------------------------------#

loss, mae, mse = model.evaluate(features_test, labels_test, verbose=0)

print("Testing set Mean Abs Error: {:5.2f} Quality".format(mae))

#-------------------------  Predicting Quality (Labels) from test data -----------------------------#

test_predictions = model.predict(features_test)

#--------------------------  Plotting and Comparing predictions with true Labels --------------------#

plt.figure()
plt.scatter(labels_test, test_predictions)
plt.xlabel('True Values [Quality]')
plt.ylabel('Predictions [Quality]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,10])
plt.ylim([0,10])
_= plt.plot([-100,10],[-100,10], c = 'r')
plt.title(str(EPOCHS) + ' ' + 'Epochs')