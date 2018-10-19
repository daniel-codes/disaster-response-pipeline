# Imports
import sys
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.corpus import stopwords
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    ''' Loads sqlite database and return X (training) and y (target) arrays
    and associated y category names
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disasterdata', engine)
    
    X = df['message'].values
    y = df.drop(['id', 'message', 'original','genre'], axis=1).values
    
    category_names = df.drop(['id', 'message', 'original','genre'], axis=1)
    
    return X, y, category_names


def tokenize(text):
    ''' Defines tokenizer for the sklearn CountVectorizer feature
    extraction model which converts a collection of text documents 
    to a matrix of token (word) counts. Returns list of clean tokens.
    '''
    
    # Normalize and lower case text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)
    
    # Remove stop words
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        # lemmatize and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    ''' Defines and returns the pipeline for the classifier model.
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=200, n_jobs=-1)))
    ])
    
    # Grid CV search parameters 
    # Note: other combinations were tested sequentially due to massive runtime,
    # therefore, commenting out prior tested parameters. Default values tended 
    # to give the best results
    parameters = {
    #'vect__max_df': (0.75, 1.0),    
    #'vect__ngram_range': ((1, 1), (1, 3)),
    #'vect__max_features': (None, 5000, 10000),
    #'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2', None),
    #'clf__estimator__n_estimators': [50, 100, 200],
    'clf__estimator__min_samples_split': [2, 4],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=1)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    ''' Evaluates the performance of the pipeline model for each target
    category using the sklearn classification_report function. Also prints
    the average precision, recall and f1 score for all categories. 
    '''
    # Lists for storing the precision, recall and f1 scores
    prec_values, recall_values, f1_values = [], [], []

    # Run prediction using the fitted model
    Y_pred = model.predict(X_test)
    
    # Insure prediction and test arrays are the same size
    assert Y_pred.shape == Y_test.shape,"Prediction and test arrays not same size!"
    
    for i in range(0, Y_pred.shape[1]):
        print('-'*40)
        print('CATEGORY = {}'.format(category_names.columns[i]))
        print('-'*40)
        cls_rpt = classification_report(Y_test[:,i][np.newaxis].T, 
                                        Y_pred[:,i][np.newaxis].T)
        print(len(cls_rpt.split()))
        
        # Track metrics for all columns
        if len(cls_rpt.split()) > 20:
            prec_values.append(float(cls_rpt.split()[17]))
            recall_values.append(float(cls_rpt.split()[18]))
            f1_values.append(float(cls_rpt.split()[19]))
    
        print('For all columns ---------')
        print('Average precision: {:.2f}'.format(np.array(prec_values).mean()))
        print('Average recall: {:.2f}'.format(np.array(recall_values).mean()))
        print('Average f1 values: {:.2f}'.format(np.array(f1_values).mean()))
    return


def save_model(model, model_filepath):
    ''' Save the fitted model '''
    pickle.dump(model, open(model_filepath, 'wb'))
    return


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
        
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