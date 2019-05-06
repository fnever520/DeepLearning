from flask import Flask, render_template, request, jsonify
from wtforms import SubmitField, Form, TextAreaField
from keras.models import load_model
from keras.datasets import imdb
import dataprocessing as dp
import tensorflow as tf

import requests


# get LSTM model and data for estimation
model = load_model('imdb_lstm_model')
graph = tf.get_default_graph()
top_words = 4500
wordToIndex = imdb.get_word_index()

# flask application
app = Flask(__name__)


# form class
class NameForm(Form):
    textArea = TextAreaField()
    submit = SubmitField('Press to get estimation')


# estimation via form on html page
@app.route('/', methods=['GET', 'POST'])
def index():
    review_estimation = "Please enter your review:"
    form = NameForm()

    if request.method == "POST":
        # get result using REST
        data = request.form['textArea']
        url = 'http://0.0.0.0:5000/rest'
        parameters = {'review': data}
        response = requests.get(url, params=parameters)
        review_estimation = response.json()['review estimation']
        form.textArea.data = data
    return render_template('index.html', form=form, name=review_estimation)


# estimation using rest service
@app.route('/rest', methods=['GET', 'POST'])
def rest():
    review = ''
    if request.method == 'GET':
        review = request.args['review']
    elif request.method == 'POST':
        review = request.form['review']
    if review != '':
        # Required because of a bug in Keras when using tensorflow graph cross threads
        with graph.as_default():
            # get sentiment estimation from model
            review_estimation = query(review)
            data = {'review estimation': review_estimation}
            return jsonify(data)


# get sentiment estimation from model
def query(text):
    # prepare input text
    data_prep = dp.PrepareData(is_one_file=True)
    input_data = data_prep.get_data(is_text=True, text=text)
    input_data = data_prep.review_truncate(input_data)
    # query
    model_test_res = model.predict(input_data)
    model_test_res = round(model_test_res[0][0])  # to get 0 or 1 only
    # result
    if model_test_res == 0:
        return "This review is negative :("
    else:
        return "This review is positive :)"


# run
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

