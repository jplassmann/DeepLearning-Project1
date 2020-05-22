"""IMPORT + SETTINGS"""

import torch

from torch import nn
from torch.nn import functional as F
from torch import optim
from torchvision import datasets
from statistics import mean
from statistics import stdev

import dlc_practical_prologue as prologue
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
import time

import models

torch.manual_seed(0)

def generate_dataset():
    """ Computation of the number of wrong classifications.
    
        Returns:
            train_input:          Tensor of size N x 2 x 14 x 14 representing the input dataset
            train_target:         Tensor of size N x 1 representing the target of the dataset
            test_input:           Tensor of size N x 2 x 14 x 14 representing the input dataset
            test_target:          Tensor of size N x 1 representing the target of the dataset
            train_classes:        Tensor of size N x 2 x 10 representing the class of the two handwritten digits

    """
    train_input, train_target, train_classes, test_input, test_target, test_classes  = prologue.generate_pair_sets(1000)
    train_target.unsqueeze_(1); test_target.unsqueeze_(1)
    return train_input, train_target, test_input, test_target, train_classes

def compute_nb_errors(model, data_input, data_target,batch_size=50):
    """ Computation of the number of wrong classifications.
    
         Args:
            model:       PyTorch neural network model
            data_input:  Tensor of size N x D representing the input dataset
            data_target: Tensor of size N x 1 representing the target of the dataset
            batch_size:  Size of the batch

        Returns:
            Number of wrong classified sample.
    """
    nb_data_errors = 0
    
    for inputs, targets in zip(data_input.split(batch_size), data_target.split(batch_size)):
        output = model(inputs)
        output = output.narrow(dim=1,start=0,length=1)
        output = torch.ge(output,0.5).float()
        for k in range(len(targets)):
            if output[k] != targets[k]:
                nb_data_errors += 1
                
    return nb_data_errors

def train_model(model, train_input, train_target, test_input, test_target,
                train_classes=None, use_auxiliary_losses=False,
                round=0, epochs=30,eta=0.4,batch_size=100):
    """ Train function for model 1 and 2
    
         Args:
            model:                PyTorch neural network model
            train_input:          Tensor of size N x 2 x 14 x 14 representing the input dataset
            train_target:         Tensor of size N x 1 representing the target of the dataset
            test_input:           Tensor of size N x 2 x 14 x 14 representing the input dataset
            test_target:          Tensor of size N x 1 representing the target of the dataset
            train_classes:        Tensor of size N x 2 x 10 representing the class of the two handwritten digits
            use_auxiliary_losses: Boolean representing the use of the auxiliary_losses
            round:                Number of round
            epochs:               Number of epochs
            eta:                  Learning rate
            batch_size:           Size of the batch

        Returns:
            test_accuracy:        Accuracy on the test dataset, array of size epochs
    """
    test_accuracy = [0] * epochs
    
    # definition of the loss and optimizer
    criterion = nn.BCELoss(reduction='mean')
    auxiliary_criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=eta)
    
    # normalization
    mu, std = train_input.mean(), train_input.std()
    train_input.sub_(mu).div_(std)
    test_input.sub_(mu).div_(std)
    
    
    for i in range(epochs):
        for inputs, targets, class_targets in zip(train_input.split(batch_size),
                                                   train_target.split(batch_size),
                                                   train_classes.split(batch_size)):
            output = model(inputs) 
 
            # Prediction of which digit is larger
            loss = criterion(output.narrow(dim=1,start=0,length=1), targets.float())
        
            # Auxiliary losses for prediciting the actual digits
            if use_auxiliary_losses:
                l_2 = auxiliary_criterion(output.narrow(dim=1,start=1,length=10),
                                      class_targets[:,0])
                l_3 = auxiliary_criterion(output.narrow(dim=1,start=11,length=10),
                                      class_targets[:,1])
                loss += l_2 + l_3
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()   
        test_accuracy[i] = compute_nb_errors(model, test_input, test_target)
        train_accuracy = compute_nb_errors(model, train_input, train_target)
        test_accuracy[i] = 100 * (1 - test_accuracy[i] / test_input.size(0))
        train_accuracy = 100 * (1 - train_accuracy / train_input.size(0))
        if (round==0):
            print(f"Epoch # {i+1} / Train Accuracy [%]: {train_accuracy:.2f} / Test Accuracy [%]: {test_accuracy[i]:.2f}")
        
    if (round>0):
        print(f"Round # {round} / Train Accuracy [%]: {train_accuracy:.2f} / Test Accuracy [%]: {test_accuracy[-1]:.2f}")

    return test_accuracy;

"""START OF TEST PROGRAM"""

if __name__=="__main__":

    print("")
    print("Mini Project 1 - Deep Learning (EE-559) - PILLONEL Ken (270852) - PLASSMANN Jeremy (2739081)")
    print("")
    print("We recommend using default values by pressing the ENTER key.")
    print("")
    #We do not do sophisticated input validation because it is not the goal of this exercise
    choice = int(input("Please choose to run models over one round (0) or multiple rounds (1) [DEFAULT = 1] :") or "1")

    if choice == 0:

        nb_epoch = 30
        nb_epoch = int(input("Please input how many epochs you want to train over [DEFAULT = 30] :") or "30")
        print("")

        train_input, train_target, test_input, test_target, train_classes = generate_dataset()
        # First model, simplest one
        net1 = models.Net1()
        print("*** Testing Model 1 ***")
        _ = train_model(net1, train_input, train_target, test_input, test_target, train_classes, epochs=nb_epoch)
        print("")
        # Second model introduces weight sharing for the convolutional layer
        net2 = models.Net2()
        print("*** Testing Model 2 ***")
        _ = train_model(net2, train_input, train_target, test_input, test_target, train_classes, epochs=nb_epoch)
        print("")
        # Third model, we use the label of the digits as an auxiliary loss
        net3 = models.Net3()
        print("*** Testing Model 3 ***")
        _ = train_model(net3, train_input, train_target, test_input, test_target, train_classes=train_classes,
                        use_auxiliary_losses=True, epochs=nb_epoch)
        print("")
        # Fourth model, we add batch normalization
        net4 = models.Net4()
        print("*** Testing Model 4 ***")
        _ = train_model(net4, train_input, train_target, test_input, test_target, train_classes=train_classes,
                        use_auxiliary_losses=True, epochs=nb_epoch)

    else:

        nb_epoch = int(input("Please input how many epochs you want to train over [DEFAULT = 30] :") or "30")
        nb_round = int(input("Please input how many rounds of training you want to perform [DEFAULT = 10] :") or "10")
        print("")

        test_accuracy_1 = []
        test_accuracy_2 = []
        test_accuracy_3 = []
        test_accuracy_4 = []

        print("*** Testing Model 1 ***")
        t0 = time.perf_counter()
        for i in range(0, nb_round):
            train_input, train_target, test_input, test_target, train_classes = generate_dataset()
            net1 = models.Net1()
            test_accuracy_1.append(train_model(net1, train_input, train_target, test_input,
                                             test_target, train_classes, round=i+1,epochs=nb_epoch))
        t_tot = time.perf_counter() - t0
        print(f"Mean Test Accuracy [%]: {mean([row[-1] for row in test_accuracy_1]):.2f} / STD [%]: {stdev([row[-1] for row in test_accuracy_1]):.2f} / Total Time [s]: {t_tot:.2f} / Mean Time [s]: {t_tot/nb_round:.2f}")
        print("")  

        print("*** Testing Model 2 ***")
        t0 = time.perf_counter()
        for i in range(0, nb_round):
            train_input, train_target, test_input, test_target, train_classes = generate_dataset()
            net2 = models.Net2()
            test_accuracy_2.append(train_model(net2, train_input, train_target, test_input,
                                             test_target, train_classes, round=i+1,epochs=nb_epoch))
        t_tot = time.perf_counter() - t0
        print(f"Mean Test Accuracy [%]: {mean([row[-1] for row in test_accuracy_2]):.2f} / STD [%]: {stdev([row[-1] for row in test_accuracy_2]):.2f} / Total Time [s]: {t_tot:.2f} / Mean Time [s]: {t_tot/nb_round:.2f}")
        print("")  

        print("*** Testing Model 3 ***")
        t0 = time.perf_counter()
        for i in range(0, nb_round):
            train_input, train_target, test_input, test_target, train_classes = generate_dataset()
            net3 = models.Net3()
            test_accuracy_3.append(train_model(net3, train_input, train_target, test_input,
                                                test_target, train_classes=train_classes, use_auxiliary_losses=True,
                                                round=i+1,epochs=nb_epoch))
        t_tot = time.perf_counter() - t0
        print(f"Mean Test Accuracy [%]: {mean([row[-1] for row in test_accuracy_3]):.2f} / STD [%]: {stdev([row[-1] for row in test_accuracy_3]):.2f} / Total Time [s]: {t_tot:.2f} / Mean Time [s]: {t_tot/nb_round:.2f}")
        print("")  

        print("*** Testing Model 4 ***")
        t0 = time.perf_counter()
        for i in range(0, nb_round):
            train_input, train_target, test_input, test_target, train_classes = generate_dataset()
            net4 = models.Net4()
            test_accuracy_4.append(train_model(net4, train_input, train_target, test_input,
                                                test_target, train_classes=train_classes, use_auxiliary_losses=True,
                                                round=i+1,epochs=nb_epoch))
        t_tot = time.perf_counter() - t0
        print(f"Mean Test Accuracy [%]: {mean([row[-1] for row in test_accuracy_4]):.2f} / STD [%]: {stdev([row[-1] for row in test_accuracy_4]):.2f} / Total Time [s]: {t_tot:.2f} / Mean Time [s]: {t_tot/nb_round:.2f}")
        print("")  

        choice_plot = int(input("Do you want to plot the results, YES = 1, NO = 0 [DEFAULT = 1] :") or "1")

        if choice_plot == 1:

            mean_1 = [0] * nb_epoch
            std_1 = [0] * nb_epoch
            mean_2 = [0] * nb_epoch
            std_2 = [0] * nb_epoch
            mean_3 = [0] * nb_epoch
            std_3 = [0] * nb_epoch
            mean_4 = [0] * nb_epoch
            std_4 = [0] * nb_epoch

            for i in range(nb_epoch):
                mean_1[i] = mean([row[i] for row in test_accuracy_1])
                std_1[i] = stdev([row[i] for row in test_accuracy_1])
                mean_2[i] = mean([row[i] for row in test_accuracy_2])
                std_2[i] = stdev([row[i] for row in test_accuracy_2])
                mean_3[i] = mean([row[i] for row in test_accuracy_3])
                std_3[i] = stdev([row[i] for row in test_accuracy_3])
                mean_4[i] = mean([row[i] for row in test_accuracy_4])
                std_4[i] = stdev([row[i] for row in test_accuracy_4])

            x = list(range(1,nb_epoch+1)) 

            fig, ax = plt.subplots(facecolor='w')
            plt.grid()
            ax.plot(x,mean_1, color='b', label='Model 1')
            ax.fill_between(x, ([x1 - x2 for (x1, x2) in zip(mean_1, std_1)]),
                            ([x1 + x2 for (x1, x2) in zip(mean_1, std_1)]), color='b', alpha=.1)

            ax.plot(x,mean_2, color='r', label='Model 2')
            ax.fill_between(x, ([x1 - x2 for (x1, x2) in zip(mean_2, std_2)]), 
                            ([x1 + x2 for (x1, x2) in zip(mean_2, std_2)]), color='r', alpha=.1)

            ax.plot(x,mean_3, color='g', label='Model 3')
            ax.fill_between(x, ([x1 - x2 for (x1, x2) in zip(mean_3, std_3)]), 
                            ([x1 + x2 for (x1, x2) in zip(mean_3, std_3)]), color='g', alpha=.1)

            ax.plot(x,mean_4, color='m', label='Model 4')
            ax.fill_between(x, ([x1 - x2 for (x1, x2) in zip(mean_4, std_4)]), 
                            ([x1 + x2 for (x1, x2) in zip(mean_4, std_4)]), color='m', alpha=.1)

            plt.xlabel('Epoch Number', fontsize=20)
            plt.ylabel('Test Accuracy [%]', fontsize=20)
            ax.legend(prop={'size': 20})
            fig.set_size_inches(18.5, 10.5)
            fig.set_dpi(100)
            plt.savefig("plot.pdf",bbox_inches='tight')
            print("")
            print(f"The plot was successfully saved in the same directory as 'test.py', it is called 'plot.pdf'.")
            print(f"The plot represents the means for each epoch with their standard deviation over {nb_round:.0f} rounds.")

    print("")
    print("*** End of file, thank you ! ***")











