from Interface import Interface
from ML.ML_trainAlgo import Trainer

import joblib



if __name__ == '__main__':
  
    interface=Interface()
 
    # I- Classifier
    result_classifier=interface.predict_input_classifier()
    
    with open('result_classifier.txt', 'w') as f:
        f.write(str("species\n"))
        f.write(str(result_classifier))
    
    # II- Regressor
    result_regressor=interface.predict_input_regressor()

    with open('result_regressor.txt', 'w') as f:
        f.write(str("species\n"))
        f.write(str(result_regressor))