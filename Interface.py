import pandas as pd 

from ML.ML_trainAlgo import Trainer



class Interface():
    
    def read_input_classifier(self):
        return pd.read_csv("input_classifier.txt",delimiter=";",header=3)
    
    def predict_input_classifier(self):
        
        tree=Trainer.load_classifier()
        return tree.predict(self.read_input_classifier())
    
    def read_input_regressor(self):
        return pd.read_csv("input_regressor.txt",delimiter=";",header=3)
    
    def predict_input_regressor(self):
        
        tree=Trainer.load_regressor()
        return tree.predict(self.read_input_regressor())

