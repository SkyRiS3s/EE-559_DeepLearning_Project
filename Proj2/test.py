from modules import *
from utils import *


def main():

    torch.manual_seed(0)

    #Generating dataset
    train_set, train_labels = generate_dataset(1000)
    test_set, test_labels = generate_dataset(1000)

    #Defining lr for Linear
    lr = 1e-6

    test_acc = []
    for i in range(10):
        print("Round " + str(i+1))

        #model
        model = Sequential([Linear(2, 25, lr), Tanh(), Linear(
            25, 25, lr), Tanh(), Linear(25, 25, lr), Tanh(), Linear(25, 2, lr), Sigmoid()])

        #Training Model
        print("*************** TRAINING ***************")
        _, training_losses, test_losses = train(
            model, train_set, train_labels, 1200, 25, MSE_loss, test_set, test_labels)
        train_errs = get_errors(model, train_set, train_labels)
        train_metric = get_metrics(model, train_set, train_labels)
        print("****************************************")
        print("Train set : \t" + "n_err = " + str(train_errs) + "\t Accuracy : " + str(train_metric[0]) + "\t Error rate : " + str(train_metric[1]))
        test_errs = get_errors(model, test_set, test_labels)
        test_metric = get_metrics(model, test_set, test_labels)
        print("Test set :  \t" + "n_err = " + str(test_errs) + "\t Accuracy : " + str(test_metric[0]) + "\t Error rate : " + str(test_metric[1]) + "\n")
        test_acc.append(test_metric[0])
        
        print("********************************************************************************")

    print("Average test accuracy : " + str(sum(test_acc)/len(test_acc)))

if __name__ == "__main__":
    main()

