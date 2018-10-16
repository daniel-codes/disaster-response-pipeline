# Import Statements
import sys
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
import numpy as np

def load_data(messages_filepath, categories_filepath):
    ''' Loads messages and associated message categories into DataFrames.
    Assumes the input files are csv formated. Merges both DataFrames and 
    returns the merged DataFrame. 
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, left_on='id', right_on='id', how='inner')
    return df

def clean_data(df):
    ''' This function takes a DataFrame with merged message/category columns
    and performs the following:
    1. Renames the column names
    2. Formats the category series as integer 1 or 0
    3. Removes duplicate rows
    4. Returns a clean DataFrame
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = [i[:-2] for i in row]
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.extract('\w*[-](\d)', expand=False)
        # convert column from string to numeric
        categories[column] = categories[column].astype('int')
    
    # Insure all category values are only 1 or 0
    for column in categories:
        a = np.array(categories[column].values.tolist())
        categories[column] = np.where(a > 1, 1, a).tolist()
        
    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)    
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicate rows
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filename):
    ''' Store cleaned DataFrame in a sqlite database with a name defined by
    the user (database_filename)
    '''
    sql_engine = create_engine('sqlite:///' + str(database_filename))
    df.to_sql('disasterdata', sql_engine, index=False, if_exists='replace')
    return 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()