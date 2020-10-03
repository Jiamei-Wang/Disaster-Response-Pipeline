# Disaster Response Pipeline Project

# Project-1-An-Analysis-on-Data-Analyst-Jobs
## Table of Contents

1. [Installation](#Installation)
2. [Project Motivation](#Project-Motivation)
3. [File Description](#File-Description)
4. [Instructions](#Instructions)
5. [Results](#Results)
6. [Acknowledgements](#Acknowledgements)

## Installation
There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python. 

## Project Motivation
The purpose of the project is to build a model for an API that classifies disaster messages and to create a web app where an emergency worker can input a new message and get classification results.

## File Description

1. **"data" Folder**
    * _disaster_categories.csv_: dataset including all the categories
    * _disaster_messages.csv_: data set containing real messages that were sent during disaster events
    * _process_data.py_: a data cleaning pipeline that extracts data from both CSV files and saves the merged and cleaned data to a SQLite database.
    * _DisasterResponse.db_: SQLite database generated from "process_data.py".

2. **"models" Folder**
    * _train_classifier_.py: Machine learning pipeline that loads data, trains and tunes a model and saves the trained model as a pickle file
    * _classifier.pkl_: Output of "train_classifier.py"

3. **"app" Folder**
    * _run.py_: Flask file to run the web application that visualizes the results
    * _templates_: contains html file for the web applicatin

3. **ETL/ ML Pipeline Preparation.jpynb**
   * Development process of the ETL/ ML Pipeline.
   
## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py` 

3. Go to http://0.0.0.0:3001/ (or try http://localhost:3001 in a browser if it does not work)

## Results
1. The ETL pipleline cleans and merges two data sets and saves the output into a SQLite database used for the machine learning pipeline.
2. The machine learning pipeline trains a multi-output classifier on the 36 categories in the dataset.
3. The Flask app to visualizes data and classifies messages entered on the web page.
      
## Acknowledgements
Credits must be given to Udacity for the code templates and FigureEight for providing the source data.
















