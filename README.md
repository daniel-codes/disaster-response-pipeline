# Disaster Response Pipeline Project

This project demonstrates the use of extract, transform, and load (ETL) and machine learning (ML) pipelipe techniques for use in a real life application. The resulting web app is targeted for emergency workers who can input new messages and get classification results in several categories (e.g. weather, medical, infrastructure, etc.). 

## Getting Started

Cloning the git repository and installing the provided packages will help you get a copy of the project up and running on your local machine. The analysis for this project was performed using Jupyter Notebook (.ipynb) and the packages were managed using the Ananconda platform. 

```
git clone https://github.com/daniel-codes/disaster-response-pipeline.git
pip install -r /path/to/requirements.txt
```

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Details:

##### ETL Pipeline
The `process_data.py` file is a data cleaning pipeline that:

* Loads the messages and categories datasets  
* Merges the two datasets    
* Cleans the data  
* Stores it in a SQLite database  

##### ML Pipeline
The `train_classifier.py` file is a machine learning pipeline that:

* Loads data from the SQLite database
* Splits the dataset into training and test sets
* Builds a text processing and machine learning pipeline
* Trains and tunes a model using GridSearchCV
* Outputs results on the test set
* Exports the final model as a pickle file

## Authors

- **Daniel Cummings** - [daniel-codes](https://github.com/daniel-codes)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

The project was done as part of the Udacity Data Science Nanodegree program. 
