"""run.py:"""
#!/usr/bin/env python
import torch    
import torch.distributed as dist
from torch.multiprocessing import Process
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms

import numpy as np
import sys, time, math, random


# Get iterative objects for MNIST data using torchvision
def load_data(batch_size):
    torch.manual_seed(0)
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

# Function to simulate stragglers
def straggler():
    shut = random.random()
    if(shut > 0.98):
        time.sleep(.001)


def layer(W, x, assignments, drop, m, n):
    """
    This function implements a linear layer with dropout in a distributed
    environment. There are three different roles in this framework. The master
    sends data to the other nodes and recieves the final result from the 
    pseudo master. The pseudo master receives matrix computations from the
    slaves and sends the fasted results to the master. The slaves each 
    compute their section of the W matrix multiplied by the x matrix.
    """

    # Shuffle assignments to avoid slow nodes getting same role each time
    np.random.shuffle(assignments[1:])
    master = assignments[0]
    pseudo_master = assignments[1]
    slaves = assignments[2:]

    # Geth the number of slaves and number parts to drop for dropout
    num_slaves = len(slaves)
    num_recv = int((1 - drop)*num_slaves)

    # Get size of the x matrix and create some placeholders for slaves
    _, p = x.size()
    W_s = torch.zeros(m//num_slaves, n)
    shards = []

    # Index list used for shuffling so same nodes not assigned same part of W each time
    index_list = np.arange(m)
    np.random.shuffle(index_list)

    # Master splits up the W matrix
    if(dist.get_rank() == master):
        shards = [W_s] + [W[index_list[m//num_slaves*i:m//num_slaves*(i+1)]] for i in range(num_slaves)] 

    # Master sends piece of W matrix to each slave
    group = dist.new_group([master] + list(slaves))
    dist.scatter(W_s, shards, master, group)
    dist.broadcast(x, master)
    dist.destroy_process_group(group)

    # Master waits for final result and returns it
    if(dist.get_rank() == master):
        y = torch.zeros(m, p)
        req = dist.irecv(tensor=y, src=pseudo_master)
        req.wait()
        return y

    # Pseudo master collects computations from slaves, sends fastest results to master
    elif(dist.get_rank() == pseudo_master):
        y1 = torch.zeros(m, p)
        y2 = torch.zeros(m, p)

        # Recieve computations from slaves in y1 tensor
        reqPieces = []
        for i in range(num_slaves):
            reqPieces.append(dist.irecv(tensor=y1[m//num_slaves*i : m//num_slaves*(i+1)], src=slaves[i]))

        # Collect fasted results and save them in the y2 tensor
        count = 0
        recvd = [0]*num_slaves
        while(count < num_recv):
            for i in range(num_slaves):
                # When a request is completed, bump up count
                if(reqPieces[i].is_completed() and recvd[i]==0):
                    count = count + 1
                    recvd[i] = 1
                    y2[index_list[m//num_slaves*i:m//num_slaves*(i+1)]] = y1[m//num_slaves*i:m//num_slaves*(i+1)]
                # When count is high enough, we have the completed computation
                if(count == num_recv):                    
                    break

        # Send result to master
        req = dist.isend(tensor=y2, dst=master)
        req.wait()

        # Wait for remaining slaves to finish and recieve results
        for i in range(num_slaves):
            reqPieces[i].wait()

    # Slaves compute the matrix multiplication for thier piece of W1 and x
    elif(dist.get_rank() in slaves):
        # Random chance this is a straggler
        straggler()

        # Run computation
        out = torch.zeros(m//num_slaves, p)
        out[:] = torch.matmul(W_s, x)/(1 - drop)
        req = dist.isend(tensor=out, dst=pseudo_master)
        req.wait()


def net():
    """
    This function creates, trains and tests our model with the MNIST dataset.
    The model has one fully connected hidden layer with m units and it uses 
    relu as an activation function. The output layer is a fully connected linear
    layer with ten output units and we apply a log softmax to get probabilities
    for each class in the data.

    We train the model over a user defined number of epochs and test the data 
    on the MNIST test data to recieve an accuracy. We also time the training 
    to compare how fast the model trains with different parameters.

    The layer function runs a pass through a layer distributedly user our 
    master/pseudo master/slave architecture. It applies a user defined amount
    of dropout as well.
    """
    
    # Random seed necessary to sync up all machines
    np.random.seed(0)

    # Set some parameters
    m = 50000 #1000
    n = 784  #100
    num_output = 10
    batch_size = 64
    batch_num = 900
    epochs = 5
    num_slaves = dist.get_world_size() - 2

    master = 0
    learning_rate = 1e-2
    dropout = 0.2

    # Assignments are used to choose which machine is pseudo master vs slaves
    assignments = np.arange(0, 2 + num_slaves)

    # The master loads up the data
    if(dist.get_rank() == 0):
        train_loader, test_loader = load_data(batch_size)

        # Initialize random W1, W2
        W1 = torch.randn(m, n)
        W2 = torch.randn(num_output, m)
        
        # Begin the timer after the data is loaded
        start = time.time()

    # Other workers create dummy variables
    else:
        W1 = None
        W2_t = None
        x = torch.empty(n, batch_size)
        grad_y_pred = torch.empty(num_output, batch_size)
        train_loader = [[x, []]] * batch_num
        test_loader = [[x, []]] * 150
        
    # One epoch for now
    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader):       

            # End when the batch is finished
            if i >= batch_num:
                break

            # Set x to the right shape and run first layer
            x = x.view(-1, 784)
            x = x.t().contiguous()
            h = layer(W1, x, assignments, dropout, m, n)

            if(dist.get_rank() == 0):
                h_relu = h.clamp(min=0)

                y_pred = W2.mm(h_relu)
                y_pred.requires_grad_(True)
                y_soft = (F.log_softmax(y_pred, dim=0)).t()

                # Compute and print loss
                loss = F.nll_loss(y_soft, y)
                if i % 300 == 0:
                    print(epoch, i, loss.item())

                # Backprop to compute gradients of w1 and w2 with respect to loss
                loss.backward()
                grad_y_pred = y_pred.grad
                grad_w2 = grad_y_pred.mm(h_relu.t()) 

                #grad_h_relu = W2.t().mm(grad_y_pred)
                W2_t = W2.t()

            grad_h_relu = layer(W2_t, grad_y_pred, assignments, 0, m, num_output)
            
            if(dist.get_rank() == 0):
                grad_h = grad_h_relu.clone()
                grad_h[h <= 0] = 0
                grad_w1 = grad_h.mm(x.t())

                # Update weights using gradient descent
                W1 -= learning_rate * grad_w1
                W2 -= learning_rate * grad_w2
                
    # Print the training time
    if dist.get_rank() ==  0:
        end = time.time()
        print('Total Time = {}'.format(end - start))
        
    # Test the model
    test_loss = 0
    correct = 0
    num_preds = 0
    for i, (x, y) in enumerate(test_loader):

        if i >= 150:
            break
        
        # Set x to the right shape and run first layer
        x = x.view(-1, 784)
        x = x.t().contiguous()
        h = layer(W1, x, assignments, 0, m, n)

        # Run second layer with the master
        if(dist.get_rank() == 0):
            h_relu = h.clamp(min=0)
            y_pred = W2.mm(h_relu)
            output = (F.log_softmax(y_pred, dim=0)).t()
        
            test_loss += F.nll_loss(output, y, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(y.view_as(pred)).sum().item()
            num_preds += len(y)
            test_loss /= batch_num
        
    if dist.get_rank() == 0:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, num_preds,
            100. * correct / num_preds))
        

def init_processes(fn, backend='mpi'):
    dist.init_process_group(init_method='env://', backend='mpi')
    fn() 
    

if __name__ == "__main__":
    init_processes(net, backend='mpi')
