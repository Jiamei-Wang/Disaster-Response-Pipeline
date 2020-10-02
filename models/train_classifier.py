import sys
import pandas as pd
import pickle
from sqlalchemy import create_engine

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

def load_data(database_filepath):
    '''load cleaned data generated from ETL Pipeline
    INPUT:
        database_filepath -- File path of sqlite database
        
    OUTPUT:
        X -- Disaster message data (features)
        Y -- 36 Categories (target)
        category_names -- Labels for 36 categories
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponseTable', engine)
    X = df.message
    Y = df.iloc[:, 4:]
    category_names = list(df.columns[4:])

    return X, Y, category_names


def tokenize(text):
    '''
    INPUT 
        text -- Original text   
    OUTPUT
        Returns a processed text variable that was tokenized, lower cased,  stripped, and lemmatized
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

    
def build_model():
    '''Build Machine Learning Pipeline
    INPUT: None
    OUTPUT:
         model --  grid search model with pipeline and classifier
    '''
    # model pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # define parameters for GridSearchCV
    parameters = {'clf__estimator__max_depth': [10, 50],
                  'clf__estimator__min_samples_leaf': [2, 5]}
                 
    # create GridSearch object and return as final model pipeline
    cv = GridSearchCV(pipeline, parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''Evaluate model performance 
    INPUT:
        model --model to be evaluated
        X_test -- test data
        y_test -- true labels for test data
        category_names -- Labels for 36 categories
    OUTPUT:
        Print the f1 score, precision and recall for each output category
    '''
    Y_pred = model.predict(X_test)
    
    # Calculate the accuracy for each of them.
    for i in range(len(category_names)):
        print("------------------------------------------------------\n",
              "Category: {}".format(category_names[i]),
              "\n", 
              classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))



def save_model(model, model_filepath):
    '''Export model as a pickle file 
    INPUT: 
        model -- Model to be saved
        model_filepath -- path of the output pick file
    OUTPUT:
        A pickle file of the final model
    '''
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()