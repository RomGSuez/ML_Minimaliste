from Interface import Interface
from ML.ML_trainAlgo import Trainer

import joblib



if __name__ == '__main__':
    # Trainer.initialize_classifier()


    #TEST
    
    # tree=Trainer.load_classifier()
    tree=joblib.load('ML\saved_clf_tree.pkl')
    print(tree)

    interface=Interface()
    print(interface.predict_input())
    result=interface.predict_input()

    with open('result.txt', 'w') as f:
        f.write(str("species\n"))
        f.write(str(result))