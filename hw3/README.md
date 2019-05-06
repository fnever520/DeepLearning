git: https://github.com/vitkorystov/imdb_reviews_sentiment_analysis


# imdb_reviews_sentiment_analysis
LSTM model for sentiment analys of imdb's reviews with Flask backend and REST service

For testing model download and unpack dataset using link 
http://ai.stanford.edu/~amaas/data/sentiment/

1. File create_model.py create and train lastm model using imdb train dataset. Reached accurancy is 85.5%
2. File dataprocessing.py allows to prepare text files (or text only) for feeding to lstm model. 
3. File start_server.py runs flask server with page to test your review using web-form and rest service (address is 0.0.0.0:5000/rest with key=review and json feedback)
4. File rest_query.py allows to test rest service.
