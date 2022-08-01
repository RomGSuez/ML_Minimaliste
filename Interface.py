import pandas as pd 

from ML.ML_trainAlgo import Trainer



class Interface():
    
    def read_input_classifier(self):
        return pd.read_csv("input_classifier.txt",delimiter=";",header=3)
    
    def predict_input(self):
        
        tree=Trainer.load_classifier()
        return tree.predict(self.read_input())

