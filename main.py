from Interface import Interface
from ML.ML_trainAlgo import Trainer

import joblib



if __name__ == '__main__':
  
    interface=Interface()
    trainer=Trainer()
    
    joblib.dump(trainer.train_classifier, 'ML/saved_clf_tree.pkl')
    joblib.dump(trainer.train_regressor, 'ML/saved_regressor_tree.pkl')
    
    
    # I- Classifier
    result_classifier=interface.predict_input_classifier()
    
    with open('Classifier/result_classifier.txt', 'w') as f:
        f.write(str("species\n"))
        f.write(str(result_classifier))
        
        print("result_classifier",result_classifier)
    
    # II- Regressor
    result_regressor=interface.predict_input_regressor()

    with open('Regressor/result_regressor.txt', 'w') as f:
        # f.write(str("species\n"))
        f.write(str(result_regressor))
        
        print("result_regressor",result_regressor)