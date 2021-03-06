"""run.py:"""
#!/usr/bin/env python
import os
import torch    
import torch.distributed as dist
from torch.multiprocessing import Process
import socket
import numpy as np

import sys

import time
from multiprocessing import Process
import pdb
import math
import random

import torch.nn.functional as F
import torch.nn as nn


DEBUG = False

def getIP():
    host_name = socket.gethostname() 
    host_ip = socket.gethostbyname(host_name) 
    return host_ip

def getPort():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('0.0.0.0', 0))
    return sock.getsockname()[1]

def getAddress():
    return str(getIP())+":" + str(getPort())

def straggler():
    #r = np.random.random_sample()
    #time.sleep(r)
    shut = random.random()
    if(shut > 0.9):
        #print(str(dist.get_rank())+":Failure..")
        time.sleep(.01)
        #print(str(dist.get_rank())+":Revived")


def shout(msg):
    if(DEBUG):
        print("Node " + str(dist.get_rank()) + " " + str(getAddress()) + " " + str(msg))

def layer(W, x, assignments, drop, m, n):

    np.random.shuffle(assignments[1:])
    master = assignments[0]
    pseudo_master = assignments[1]
    slaves = assignments[2:]

    num_slaves = len(slaves)
    num_recv = int((1 - drop)*num_slaves)

    _, p = x.size()

    W_s = torch.zeros(m//num_slaves, n)
    shards = []

    #if(dist.get_rank() in [master, pseudo_master]):
    index_list = np.arange(m)
    np.random.shuffle(index_list)

    if(dist.get_rank() == master):
        shards = [W_s] + [W[index_list[m//num_slaves*i:m//num_slaves*(i+1)]] for i in range(num_slaves)] 

    #group = dist.new_group([0] + [i for i in range(2,2+num_slaves)])
    group = dist.new_group([master] + list(slaves))
    dist.scatter(W_s, shards, master, group)
    dist.broadcast(x, master)

    if(dist.get_rank() == master):
        y = torch.zeros(m, p)

        req = dist.irecv(tensor=y, src=pseudo_master)
        req.wait()
        return y

    elif(dist.get_rank() == pseudo_master):
        y1 = torch.zeros(m, p)
        y2 = torch.zeros(m, p)

        reqPieces = []
        for i in range(num_slaves):
            reqPieces.append(dist.irecv(tensor=y1[m//num_slaves*i : m//num_slaves*(i+1)], src=slaves[i]))
            
        count = 0
        recvd = [0]*num_slaves
        while(count < num_recv):
            for i in range(num_slaves):
                if(reqPieces[i].is_completed() and recvd[i]==0):
                    count = count + 1
                    recvd[i] = 1
                    y2[index_list[m//num_slaves*i:m//num_slaves*(i+1)]] = y1[m//num_slaves*i:m//num_slaves*(i+1)]
                if(count == num_recv):                    
                    break

        req = dist.isend(tensor=y2, dst=master)
        req.wait()

        for i in range(num_slaves):
            reqPieces[i].wait()

    
    elif(dist.get_rank() in slaves):
        #straggler()
        out = torch.zeros(m//num_slaves, p)
        out[:] = torch.matmul(W_s, x)/(1 - drop)
        req = dist.isend(tensor=out, dst=pseudo_master)
        req.wait()

    else:
        pass


def net():

    np.random.seed(0)

    m = 10000 #1000
    n = 100  #100
    num_output =10
    batch_size = 64
    num_slaves = 2

    master = 0
    learning_rate = 1e-3
    dropout = 0.2

    W1 = None
    W2_t = None
    x = torch.empty(n, batch_size)
    grad_y_pred = torch.empty(num_output, batch_size)

    assignments = np.arange(0, 2 + num_slaves)

    if(dist.get_rank() == 0):
        W1 = torch.randn(m, n)
        W2 = torch.randn(num_output, m)
        x = torch.randn(n, batch_size)
        
        y = torch.zeros(batch_size, dtype=torch.long)
        for i in range(batch_size):
            y[i] = random.randint(0, num_output-1)
        
        start = time.time()
    
    for t in range(100):
        h = layer(W1, x, assignments, dropout, m, n)

        if(dist.get_rank() == 0):
            h_relu = h.clamp(min=0)

            y_pred = W2.mm(h_relu)
            y_pred.requires_grad_(True)
            y_soft = (F.log_softmax(y_pred, dim=0)).t()

            # Compute and print loss
            loss = F.nll_loss(y_soft, y)
            if t % 10 == 0:
                print(t, loss.item())

            # Backprop to compute gradients of w1 and w2 with respect to loss
            loss.backward()
            grad_y_pred = y_pred.grad
            grad_w2 = grad_y_pred.mm(h_relu.t()) 

            #grad_h_relu = W2.t().mm(grad_y_pred)
            W2_t = W2.t()

        grad_h_relu = layer(W2_t, grad_y_pred, assignments, 0, m, num_output)
            
        if(dist.get_rank() == 0):
            grad_h = grad_h_relu.clone()
            grad_h[h < 0] = 0
            grad_w1 = grad_h.mm(x.t())

            # Update weights using gradient descent
            W1 -= learning_rate * grad_w1
            W2 -= learning_rate * grad_w2

    if dist.get_rank() ==  0:
        end = time.time()
        print('Total Time = {}'.format(end - start))


def init_processes(fn, backend='mpi'):
    dist.init_process_group(init_method='env://', backend='mpi')
    fn() 
    

if __name__ == "__main__":
    init_processes(net, backend='mpi')
