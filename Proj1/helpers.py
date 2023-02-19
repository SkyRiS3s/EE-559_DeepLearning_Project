#import torch modules
import torch
from torch import nn
from torch.nn import functional
from torch import optim

#import prologue
import dlc_practical_prologue as prologue

#import our code files
from models import *

#import for plotting results
from matplotlib import pyplot as plt

def train(model, input_, target_, target_classes, epochs, mini_batch, lr, norm_weight, aux_weight):
    """
    Arguments:
        model: class model to use
        input_: input data
        target_: target data
        target_classes: target classes data
        epochs: number of epochs to train on
        mini_batch: batch size
        lr: learning rate for the optimizer
        norm_weight: normalization weight used to compute the loss
        aux_weight: auxiliary loss weight used to compute the loss

    Return:
        losses: array containing the loss for each batch in each epoch
    """
    #use Cross Entropy loss
    criterion = nn.CrossEntropyLoss()
    #use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr = lr)
    
    losses = []
    for epoch in range(epochs):
        for batch in range(0, input_.size(0), mini_batch):
            train_in = input_.narrow(0, batch, mini_batch)
            target_in = target_.narrow(0, batch, mini_batch)
            target_in_classes = target_classes.narrow(0, batch, mini_batch)
            
            #make a forward pass
            out, out_aux = model(train_in)
            #compute loss
            loss = criterion(out, target_in)
            #adjust loss to avoid overfitting
            for parameter in model.parameters():
                loss += norm_weight * parameter.pow(2).sum()
            #adjust loss if auxiliary loss is used
            if out_aux:
                loss += aux_weight*criterion(out_aux[0], target_in_classes[:,0])
                loss += aux_weight*criterion(out_aux[1], target_in_classes[:,1])
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.data.item())
            
    return losses

def get_test_error_rate(model, test_input, test_target):
    """
    Arguments:
        model: class model to use
        test_input: test input data
        test_target_: test target data

    Return:
        error rate
    """
    #make forward pass and ignore second output since only used for training
    preds,_ = model(test_input)
    preds = functional.softmax(preds, dim = 0)

    n_errs = 0
    nb_samples = test_input.size(0)
    for i in range(nb_samples):
        #choose class with maximum value
        if torch.max(preds[i], 0)[1] != test_target[i]:
            n_errs += 1

    return n_errs/nb_samples

def generate_datasets_CV(n_samples, n_CV):
    """
    Arguments:
        n_samples: number of samples to use in each dataset
        n_CV: number of datasets to generate

    Return:
        data_CV: array of datasets containing pairs of training and validation data
    """
    data_CV = []
    for i in range(n_CV):
        train_data = {}
        valid_data = {}
        train_data["input"], train_data["target"], train_data["classes"], valid_data["input"], valid_data["target"], valid_data["classes"] = prologue.generate_pair_sets(n_samples)
        train_data["input"] = normalize(train_data["input"])
        valid_data["input"] = normalize(valid_data["input"])
        data_CV.append((train_data, valid_data))

    return data_CV

def run_cross_validation(model, data_CV, n_samples, n_CV, mini_batch, lr, norm_weights, aux_weights):
    """
    Arguments:
        model: class model to use
        data_CV: datasets
        n_samples: number of samples to use in each dataset
        n_CV: number of datasets to generate
        mini_batch: batch size
        lr: learning rate for the optimizer
        norm_weights: array of normalization weights to use for CV
        aux_weights: array of auxiliary weights to use for CV
    Return:
        data_CV: array of datasets
    """
    lr = 0.01
    model = CNN_base()

    norm_weights = [0.01, 0.02, 0.05, 0.1]
    #aux_weights = [0.5, 0.75, 1, 1.25, 1.5]
    aux_weights = [0]

    score_per_model = []
    for norm_weight in norm_weights:
        for aux_weight in aux_weights:
            print("norm_weight: {0}, aux_weight: {1}".format(norm_weight, aux_weight))
            score_per_round = []
            for i in range(n_CV):
                train_data = data_CV[i][0]
                valid_data = data_CV[i][1]
                train(model, train_data["input"], train_data["target"], train_data["classes"], epochs = 25, mini_batch = mini_batch, lr = lr, norm_weight=norm_weight, aux_weight=aux_weight)
                score_per_round.append(get_test_error_rate(model, valid_data["input"], valid_data["target"]))

            score_tens = torch.Tensor(score_per_round)
            mean = torch.mean(score_tens).item()
            std = torch.std(score_tens).item()
            print(score_tens)
            score_per_model.append({"norm_weight": norm_weight, "aux_weight": aux_weight, "mean": mean, "std": std})
    
    print(score_per_model)
    return score_per_model

def normalize(data):
    """normalize data
    Arguments: data to normalize
    Return: normalized data
    """
    return (data-data.mean())/data.std()

def plot_rounds(test_accuracy, n_rounds):
    """ plot results for each round
    Arguments:
        test_accuracy: array of values to plot for each model
        n_rounds: number of rounds
    """
    plt.figure(figsize=(8, 8))
    ax = plt.axes()
    ax.plot(test_accuracy[0], label='CNN_base')
    ax.plot(test_accuracy[1], label='CNN_AL')
    ax.plot(test_accuracy[2], label='CNN_WS')
    ax.plot(test_accuracy[3], label='CNN_WS_AL')
    ax.legend()
    plt.title('Accuracy of 4 models over {} rounds'.format(n_rounds))
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.show()

def plot_means_stds(acc_means, acc_stds):
    """ plot mean and std of accuracy
    Arguments:
        acc_means: means of accuracy
        acc_stds: standard deviations of accuracy
    """
    plt.figure(figsize=(8, 8))
    plt.bar(["CNN_base", "CNN_AL", "CNN_WS", "CNN_WS_AL"], acc_means, yerr=acc_stds)
    plt.ylim([0.6, 1])
    plt.title('Mean and Standard Deviation of Accuracy')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.show()