import torch
import math
import seaborn as sns
import matplotlib.pyplot as plt


def convert_labels_to_onehot(labels):
    """Converts tensor of binary labels to one-hot"""
    return torch.stack([(~labels).add(2), labels]).T

# Circle Origin = (0.5, 0.5
# Circle equ : (x-0.5)^2+(y-0.5)^2  = 1/(2pi)
def generate_dataset(n_samples):
    """Generataes a set a coordinates (x,y) in [0,1]^2 and their labels. 
        The label for each point (x,y) determines whether the point is in a circle of origin (0.5,0.5) and of radius r = 1/sqrt(2pi)
        n_samples : number of samples to be generated"""
    rand_coordinates = torch.empty(n_samples, 2).uniform_(
        0, 1)  # generate x,y coordinates uniformally distributed
    labels = rand_coordinates.sub(0.5).pow(2).sum(dim=1).sub(
        1 / (2 * math.pi)).sign().add(1).div(2).long()
    return rand_coordinates, convert_labels_to_onehot(labels)


def plot_dataset(coordinates, labels):
    """Plots the dataset given the coordinates and their respective labels"""
    plt.subplots(figsize=(9, 6))
    sns.scatterplot(x=coordinates[:, [0]].squeeze().numpy(
    ), y=coordinates[:, [1]].squeeze().numpy(), hue=labels[:, 1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Dataset")
    plt.legend()
    plt.show()


def train(model, input_, target_, epochs, mini_batch, loss_class, test_, test_target_):
    """Trains the model"""
    losses = []
    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        for batch in range(0, input_.size(0), mini_batch):
            train_in = input_.narrow(0, batch, mini_batch)
            target_in = target_.narrow(0, batch, mini_batch)

            #Get prediction
            out = model.forward(train_in)

            #Compute loss
            loss_function = loss_class()
            loss = loss_function.forward(out, target_in)

            #Backprop
            g = loss_function.backward()
            model.backward(g)

        if (epoch+1) % 100 == 0:
            print('Epoch : ' + str(epoch+1), '\t Loss : ', loss.item())
        losses.append(loss.item())

        #Getting Train loss per epoch
        out = model.forward(input_)
        loss_function = loss_class()
        train_losses.append(loss_function.forward(out, target_).item())

        #Getting Test loss per epoch
        out = model.forward(test_)
        loss_function = loss_class()
        test_losses.append(loss_function.forward(out, test_target_).item())
        

    return losses, train_losses, test_losses


def get_errors(model, input_, target_):
    """Computes the number of prediction errors"""
    n_errs = 0
    out = model.forward(input_)
    preds = torch.max(out.data, 1)[1].data
    labels = torch.max(target_.data, 1)[1].data
    for i in range(preds.size()[0]):
        if labels[i] != preds[i]:
            n_errs += 1
    return n_errs


def get_metrics(model, input_, target_):
    """Computes the accuracy and the test error"""
    N = input_.size(0)
    errors = get_errors(model, input_, target_)
    corrects = N-errors
    return corrects/N, errors/N

def plot_loss_wrt_epochs(training_losses, test_losses):
    plt.subplots(figsize=(9, 6))
    xs = [x+1 for x in range(len(training_losses))]
    plt.plot(xs, training_losses, 'b', label = 'train')
    plt.plot(xs, test_losses, 'r', label = 'test')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title('Loss w.r.t. epochs')
    plt.legend()
    plt.show()


