from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
import dataprocessing as dp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# model parameters
top_words = 4500
review_length = 600
embedding_vector_length = 32
model_name = 'imdb_lstm_model2'
epochs_run = 10
# data preparation object
data_prep = dp.PrepareData(top_words=top_words, review_length=review_length)

#todo
def eval_metric(model, history, metric_name):
    #metric = history.history[metric_name]
    #val_metric = history.history['val_' + metric_name]
    #e = range(1, epochs_run+1)
    #plt.plot(e, metric, 'bo', label = 'Train '+ metric_name)
    print(history.history[metric_name])
    
    plt.plot(history.history[metric_name])
    #plt.plot(e, val_metric, 'b', label = 'Test ' + metric_name)
    plt.title('Model ' + metric_name)
    plt.xlabel('Epochs number')
    plt.ylabel(metric_name)
    plt.legend()
    plt.show()
    
# paths to train and test sets
path_train_pos = "aclImdb/train/pos/"
path_train_neg = "aclImdb/train/neg/"
path_test_pos = "aclImdb/test/pos/"
path_test_neg = "aclImdb/test/neg/"

# data processing
train_data_pos = data_prep.get_data(path_train_pos)
train_data_neg = data_prep.get_data(path_train_neg)

test_data_pos = data_prep.get_data(path_test_pos)
test_data_neg = data_prep.get_data(path_test_neg)

input_train, output_train = data_prep.binary_shuffle(train_data_pos, train_data_neg)
input_test, output_test = data_prep.binary_shuffle(test_data_pos, test_data_neg)

# truncate and pad input sequences
input_train = data_prep.review_truncate(input_train)
input_test = data_prep.review_truncate(input_test)

# create the model
model = Sequential()
#it computes the word embedding(or use pre-trained embeddings) and look up each word in the dict to find its vector representation. it trains the word embedding with 32 dimensions.
# the number of words here uses 4500, and running with epochhs 10
# to reduce overfitting, we could apply regularization or dropout layters - which drop certain features by setting them to zero
model.add(Embedding(top_words, embedding_vector_length, input_length=review_length))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
#compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
# train and save model
#todo
history = model.fit(input_train, output_train, epochs=epochs_run, batch_size=64, verbose = 1)
#model.fit(input_train, output_train, epochs=10, batch_size=64)
print(history.history.keys())
model.save(model_name)

#todo
eval_metric(model, history, 'loss')
eval_metric(model, history, 'acc')
# Final evaluation of the model - with the testing set
scores = model.evaluate(input_test, output_test, batch_size = 64, verbose=1)

print("i",input_test.shape)
#print("o", output_test.shape())

#added here - generates output predictions
output_prediction = model.predict(input_test, batch_size = 64, verbose = 1)
print("Accuracy: %.2f%%" % (scores[1]*100))

# added here
print("The roc AUC score is %.2f" %(roc_auc_score(output_test,output_prediction)))


'''
for gloving in particular,
due to insuffucient training dataset, the model might not be able to learn good embedding for sentiment analysis. We could use pre-trained word embedding - GloVe/ It contains multiple pre-trained word embeddings. we put word embedding in a dict where keys are the words abnd values are the word embedding,. 
with the glove embedding loaded in a dict, we can look up the embedding for each word in the corpus of the 'xx' dataset. these will be stored in a matrix with a shape of TOP_words and EMB_vector( which also known as glove dimension). if a word is not found in the glove dictionary, the word embedding values for the particular word will be zero.
'''