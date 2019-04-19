import numpy as np
import pandas as pd
from os.path import join, dirname

import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD

from sklearn.metrics import roc_auc_score
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.utils import check_X_y
from sklearn.neural_network import MLPClassifier

from collections import Counter
from imblearn.over_sampling import RandomOverSampler as OverSampler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

dataset_path = join(dirname(__file__),"HW2_data")
csv = ['BNNdata_20080701.csv', 'breast-cancer.csv', 'breast-w.csv','colic.csv', 'credit-a.csv', 'credit-g.csv', 'diabetes.csv', 'heart-statlog.csv']

csvfile = join(dataset_path,"breast-cancer.csv")

dataset = pd.read_csv(csvfile)
X = dataset.iloc[:,1:-1].values
Y = dataset.iloc[:,-1].values
del dataset['Unnamed: 0']

yes_set = 0
no_set = 0
X_train = []
X_test = []
Y_train = []
Y_test = []

for i in range(len(dataset)):
    if Y[i] == 1:
        if yes_set < 50:
            X_test.append(X[i])
            Y_test.append(Y[i])
            yes_set +=1
        else:
            X_train.append(X[i])
            Y_train.append(Y[i])
    else:
        if no_set < 50:
            X_test.append(X[i])
            Y_test.append(Y[i])
            no_set +=1
        else:
            X_train.append(X[i])
            Y_train.append(Y[i])

X_train = np.array(X_train)
print(X_train.shape)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test_ = np.array(Y_test)
Y_train = keras.utils.to_categorical(Y_train)
Y_test = keras.utils.to_categorical(Y_test_)


# print("[Before] \nThe ratio: ", Y.sum()/Y.shape[0])
print("[After]\nShape : ", Y_train.shape, Y_test.shape)
print("The ratio for testing set: ", Y_test_.sum()/Y_test.shape[0])

# HyperParameters
lr = 0.001
training_epochs = 50
batch_size = 5

# Network Parameters
n_hidden_1 = 256 # number of neurons in 1st layer 
n_hidden_2 = 9 # number of neurons in 2nd layer 
n_input = 9 
n_classes = 2  
steps = len(X_train) # How many training data
print(X_train.shape)

#Resetting the graph
tf.reset_default_graph()

#Defining Placeholders
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])
#hold_prob1 = tf.placeholder(tf.float32)
#hold_prob2 = tf.placeholder(tf.float32)

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1],mean = 0.0,stddev=0.2)),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],mean = 0.0,stddev = 0.2)),
    'out': tf.Variable(tf.truncated_normal([n_hidden_2, n_classes],mean = 0.0,stddev = 0.2))
}
biases = {
    'b1': tf.Variable(tf.constant(0.1,shape = [n_hidden_1])),
    'b2': tf.Variable(tf.constant(0.1,shape = [n_hidden_2])),
    'out': tf.Variable(tf.constant(0.1,shape = [n_classes]))
}


# Create model
def multilayer_perceptron(x):
    # First Hidden Layer
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    #Applying RELU nonlinearity
    layer1_RELU = tf.nn.relu(layer_1)
    #Applying Dropout
    layer1_dropout = tf.nn.dropout(layer1_RELU, keep_prob = 0.2)
    # Second hidden layer
    layer_2 = tf.add(tf.matmul(layer1_dropout, weights['h2']), biases['b2'])
    #Applying TANH nonlinearity
    layer2_TANH = tf.nn.tanh(layer_2)
    #Applying Dropout
    layer2_dropout = tf.nn.dropout(layer2_TANH, keep_prob = 0.2)
    # Output layer
    out_layer = tf.matmul(layer2_dropout, weights['out']) + biases['out']
    return out_layer

# Building model
logits = multilayer_perceptron(X)

# Defining Loss Function
print(X.shape)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
#Defining optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
#Defining what to minimize
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()

#Graph equations for Accuracy
matches = tf.equal(tf.argmax(logits,1),tf.argmax(Y,1))
acc = tf.reduce_mean(tf.cast(matches,tf.float32))

#Opening our Session
with tf.Session() as sess:
    #Runnig initialization of variables
    sess.run(init)
    # Writing down the for loop for epochs
    for epoch in range(training_epochs):
        #For loop for batches
        for i in range(0,steps,batch_size):
            #Getting training data to be fed into the graphs
            batch_x, batch_y = X_train[i:i+batch_size],Y_train[i:i+batch_size]
            # Training batch by batch
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
        #Feeding CV data to the graphs

        #Calculating AUC on Cross Validation data

        #Printing CV statistics after each epoch
        #print("Accuracy:",acc_on_cv,"Loss:",loss_on_cv,"AUC:",auc_on_cv)
    
    #Feeding test data to the graphs
    acc_on_test,loss_on_test,preds_on_test = sess.run([acc,loss_op,tf.nn.softmax(logits)], feed_dict = {X: X_test,Y: Y_test})
    #Calculating AUC on CV data
    auc_on_test = roc_auc_score(Y_test,preds_on_test)
    print("Test Results:")
    print("Accuracy:",acc_on_test,"Loss:",loss_on_test,"AUC:",auc_on_test)
    
    print("All done")
