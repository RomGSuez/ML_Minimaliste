import pandas as pd 

from ML.ML_trainAlgo import Trainer



class Interface():
    
    def read_input(self):
        return pd.read_csv("input.txt",delimiter=";",header=3)
    
    def predict_input(self):
        
        tree=Trainer.load_classifier()
        return tree.predict(self.read_input())
    
    def load_input_for_comparaison(self):
        df_input=read_input()

        with open('compare_feature.txt', 'w') as f:
                f.write(str("Here are the data features you are studying:\n\n"))
                f.write(str(df_input.columns)+"\n")
                # f.write(str((df.loc[1,:]).values,"\n"))
                for e in (df.loc[1,:]).values:
                        f.write(str(e)+";")

                f.write("\n\nModify here features values to see how the result evolve:\n\n")
                f.write(str(df_input.columns)+"\n")
                f.write(str("Write here your modification"))



