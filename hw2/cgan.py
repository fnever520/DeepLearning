from __future__ import print_function, division
from os.path import join, dirname
import pandas as pd
import keras
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np


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
X_test = np.array(X_test)
Y_train_ = np.array(Y_train)
Y_test_ = np.array(Y_test)
Y_train = keras.utils.to_categorical(Y_train)
print("sss",Y_train.shape)
Y_test = keras.utils.to_categorical(Y_test_)


# print("[Before] \nThe ratio: ", Y.sum()/Y.shape[0])
print("[After]\nShape : ", Y_train.shape, Y_test.shape)
print("The ratio for testing set: ", Y_test_.sum()/Y_test.shape[0])

class CGAN():
    def __init__(self):
        # Input shape
        self.rows = 1
        self.cols = 9
        self.channels = 1
        self.data_shape = (self.rows, self.cols, self.channels)
        self.num_classes = 2
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        data = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([data, label])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=['binary_crossentropy'],
            optimizer=optimizer)

    def build_generator(self):
        print("**********")
        print("Generator")
        print("**********")
        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.data_shape), activation='tanh'))
        model.add(Reshape(self.data_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        
        label = Input(shape=(1,), dtype='int32')
        
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        model_input = multiply([noise, label_embedding])
        data = model(model_input)

        return Model([noise, label], data)

    def build_discriminator(self):
        print("**********")
        print("Discriminator")
        print("**********")
        model = Sequential()

        model.add(Dense(512, input_dim=np.prod(self.data_shape)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        data = Input(shape=self.data_shape)
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.data_shape))(label))
        flat_data = Flatten()(data)

        model_input = multiply([flat_data, label_embedding])

        validity = model(model_input)

        return Model([data, label], validity)

    def train(self, x_train, y_train, epochs, batch_size=128, sample_interval=50):

        # Load the dataset

        # Configure input
        # print(x_train)
        # normalize it 
        print(x_train.shape)
        print(y_train.shape)
        x_train = (x_train.astype(np.float32) - 2.5) / 2.5
        x_train = np.expand_dims(x_train, axis=3)
        y_train = y_train.reshape(-1, 1)
        # y_train = np.expand_dims(y_train, axis=3)
        print(x_train.shape)
        print(y_train.shape)
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images

            idx = np.random.randint(0, x_train.shape[0], batch_size)
            datas, labels = x_train[idx], y_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, 100))

            # Generate a half batch of new images
            gen_datas = self.generator.predict([noise, labels])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([datas, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_datas, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on labels
            sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_data(self, epoch):
        r = 1
        noise = np.random.normal(0, 1, (r, 100))
        sampled_labels = np.arange(0, 10).reshape(-1, 1)

        gen_datas = self.generator.predict([noise, sampled_labels])

        # Rescale images 0 - 1
        gen_datas = 0.5 * gen_datas + 0.5

        return gen_datas

if __name__ == '__main__':
    cgan = CGAN()
    cgan.train(epochs=20000, x_train = X_train, y_train=Y_train_, batch_size=32, sample_interval=200)