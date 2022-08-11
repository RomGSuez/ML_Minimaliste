import numpy as np
import pandas as pd 

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import ExtraTreesRegressor

import joblib
import pickle


class Trainer:
  
        
    def train_classifier(self):
        # Load dataset
        iris = load_iris()
            
        X = iris.data
        y = iris.target
             
        # Split dataset into train and test
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size = 0.3,
                                random_state = 2018)
            
        # Create DecisionTreeClassifier model
        tree =DecisionTreeClassifier(max_depth = 10).fit(X_train, y_train)
        pickle.dump(tree, open("ML/saved_clf_tree.pkl", "wb"))
        
            
    def load_classifier(): 
        # print("load")
        return  joblib.load('ML/saved_clf_tree.pkl')
  
    
  
    
    def train_regressor(self):
        # Load dataset
        iris = load_iris()
        X = iris.data
        
        #Create new X df without the target variable (petal width)
        # Create new X df with the target variable (petal width)
        new_X=[]
        new_y=[]
         
        for arr in X:
            new_X.append(arr[:-1])
            new_y.append(arr[-1:])
          
        
        # Split dataset into train and test
        X_train, X_test, y_train, y_test = \
        train_test_split(new_X, new_y, test_size = 0.3,
                            random_state = 2018)
        
        
        # Create ExtraTreesRegressor model
        tree =ExtraTreesRegressor(max_depth = 10).fit(X_train, y_train)
        pickle.dump(tree, open("ML/saved_regressor_tree.pkl", "wb"))
        
        
        
            

    def load_regressor(): 
        # print("load")
        return  joblib.load('ML/saved_regressor_tree.pkl')
    

