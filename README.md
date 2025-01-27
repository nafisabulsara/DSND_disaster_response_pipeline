# Disaster Response Pipeline Project
This Project is part of Udacity's Data Science Nanodegree in collaboration with Figure Eight. The initial dataset contains pre-labelled tweet and messages from real-life disaster. The aim of the project is to build a Natural Language Processing tool that categorize messages.

The Project is divided in the following Sections:

Data Processing, ETL Pipeline to extract data from source, clean data and save them in a proper databse structure
Machine Learning Pipeline to train a model able to classify text message in categories
Web App to show model results in real time.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


#### Project depenencies:
Python 3.5+
NumPy
SciPy
Pandas
Scikit-Learn
NLTK
SQLalchemy
Flask
Plotly
