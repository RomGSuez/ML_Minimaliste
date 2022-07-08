import numpy as np
import pandas as pd 

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier 

import joblib

class Trainer:
       
    def initialize_classifier():

        iris=pd.read_csv("iris.csv")
        X = iris[iris.columns[iris.columns!="variety"]]
        y = iris[iris.columns[iris.columns=="variety"]]

        # Load dataset
        # iris = load_iris()
         
        # X = iris.data
        # y = iris.target
         

        # Split dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,shuffle=True)
         
        # Create DecisionTreeClassifier model
        tree = DecisionTreeClassifier(max_depth = 10).fit(X_train, y_train)

        # Save the model as a pickle in a file
        joblib.dump(tree, 'saved_clf_tree.pkl')
        
        
    
        
    def load_classifier(): 
        return  joblib.load('ML/saved_clf_tree.pkl')

    
    


