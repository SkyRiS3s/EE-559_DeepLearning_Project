#import torch modules
import torch
from torch import optim

#import prologue
import dlc_practical_prologue as prologue

#import our code files
from helpers import *
from models import *

def main():
    n_samples=1000
    n_rounds=15
    mini_batch=100
    lr = 0.01

    #-------------------  uncomment the code in this section to run cross validation  -------------------
    ## the cross validation has already been run to find the best hyperparameters which are used below
    ## results are printed and are also returned by the run_cross_validation function

    #n_CV=5
    #data_CV = generate_datasets_CV(n_samples, n_CV)
    #run_cross_validation(CNN_base, data_CV, n_samples, n_CV, mini_batch, lr, norm_weights = [0.01, 0.02, 0.05, 0.1], aux_weights = [0])
    #run_cross_validation(CNN_AL, data_CV, n_samples, n_CV, mini_batch, lr, norm_weights = [0.01, 0.02, 0.05, 0.1], aux_weights = [0.5, 0.75, 1, 1.25, 1.5])
    #run_cross_validation(CNN_WS, data_CV, n_samples, n_CV, mini_batch, lr, norm_weights = [0.01, 0.02, 0.05, 0.1], aux_weights = [0])
    #run_cross_validation(CNN_WS_AL, data_CV, n_samples, n_CV, mini_batch, lr, norm_weights = [0.01, 0.02, 0.05, 0.1], aux_weights = [0.5, 0.75, 1, 1.25, 1.5])

    #------------------------------- END of cross validation --------------------------------

    #choose models
    model_classes = [CNN_base, CNN_AL, CNN_WS, CNN_WS_AL]

    #hyperparameters for each model ordered same the way as model_classes
    norm_weights = [0.02, 0.02, 0.01, 0.01]
    aux_weights = [0, 1.25, 0, 1.25]

    test_accuracy = torch.empty((4, n_rounds))

    for i in range(n_rounds):
        train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(n_samples)
        train_input = normalize(train_input)
        test_input = normalize(test_input)

        for j, model_class in enumerate(model_classes):
            model = model_class()
            train(model, train_input, train_target, train_classes, epochs = 25, mini_batch = mini_batch, lr=lr, norm_weight=norm_weights[j], aux_weight=aux_weights[j])

            acc_round = 1-get_test_error_rate(model, test_input, test_target)
            test_accuracy[j][i] = acc_round

        print('Round ' + str(i+1))
        print('Test accuracy:\n')
        print('CNN_base: {}'.format(test_accuracy[0][i]))
        print('CNN_AL: {}'.format(test_accuracy[1][i]))
        print('CNN_WS: {}'.format(test_accuracy[2][i]))
        print('CNN_WS_AL: {}'.format(test_accuracy[3][i]))

        print('***************************************************************')

    #compute means and standard deviations accross rounds for the 4 models
    acc_means = []
    acc_stds = []
    for i in range(4):
        acc_means.append(test_accuracy[i].mean().item())
        acc_stds.append(test_accuracy[i].std().item())

    #print mean and standard deviation of results for the 4 models
    #CNN_base
    print('CNN without weight sharing and without auxiliary loss')
    print("mean of test scores: {}, std of test scores: {}".format(acc_means[0], acc_stds[0]))

    #CNN_AL
    print('CNN without weight sharing and with auxiliary loss')
    print("mean of test scores: {}, std of test scores: {}".format(acc_means[1], acc_stds[1]))

    #CNN_WS
    print('CNN with weight sharing and without auxiliary loss')
    print("mean of test scores: {}, std of test scores: {}".format(acc_means[2], acc_stds[2]))

    #CNN_WS_AL
    print('CNN with weight sharing and with auxiliary loss')
    print("mean of test scores: {}, std of test scores: {}".format(acc_means[3], acc_stds[3]))

    plot_rounds(test_accuracy, n_rounds)
    plot_means_stds(acc_means, acc_stds)

if __name__ == "__main__":
    main()