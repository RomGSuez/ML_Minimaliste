from Interface import Interface
from ML.ML_trainAlgo import Trainer

import joblib



if __name__ == '__main__':
  
    interface=Interface()
 
    result=interface.predict_input()

    with open('result.txt', 'w') as f:
        f.write(str("species\n"))
        f.write(str(result))