from Interface import Interface
from ML.ML_trainAlgo import Trainer





if __name__ == '__main__':
    Trainer.initialize_classifier()


    #TEST
    tree=Trainer.load_classifier()
    print(tree)

    interface=Interface()
    print(interface.predict_input())
    result=interface.predict_input()

    with open('result.txt', 'w') as f:
    f.write(result)