"""run.py:"""
#!/usr/bin/env python
import os
import torch    
import torch.distributed as dist
from torch.multiprocessing import Process
import socket
import numpy as np

import sys

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

def shout(msg):
    if(DEBUG):
        print("Node " + str(dist.get_rank()) + " " + str(getAddress()) + " " + str(msg))

def layer(W, x, assignments, drop):
    master = assignments[0]
    pseudo_master = assignments[1]
    slaves = assignments[2:]

    num_slaves = len(slaves)
    num_recv = int((1 - drop)*num_slaves)

    m, n = W.size()
    _, p = x.size()

    shard = int(m/num_slaves)


    if(dist.get_rank() == master):
        y = torch.zeros(m, p)

        req = dist.irecv(tensor=y, src=pseudo_master)
        req.wait()
        shout("Received top 50 percent")
        return y

    elif(dist.get_rank() == pseudo_master):
        y1 = torch.zeros(m, p)
        y2 = torch.zeros(m, p)

        reqPieces = []
        for i in range(num_slaves):
            reqPieces.append(dist.irecv(tensor=y1[shard*i : shard*(i+1)], src=slaves[i]))
        
        count = 0
        recvd = [0]*num_slaves
        while(count < num_recv):
            for i in range(num_slaves):
                if(reqPieces[i].is_completed() and recvd[i]==0):
                    count = count + 1
                    recvd[i] = 1
                    y2[shard*i:shard*(i+1)] = y1[shard*i:shard*(i+1)]
                if(count == num_recv):                    
                    break

        shout(y2)
        shout("Gathered top 50 percent "+str(recvd))
        shout("Sending top 50 percent")
        req = dist.isend(tensor=y2, dst=master)
        req.wait()

        for i in range(num_slaves):
            reqPieces[i].wait()

    
    elif(dist.get_rank() in slaves):
        index = list(slaves).index(dist.get_rank())
        out = torch.zeros(shard, p)
        out[:] = torch.matmul(W[shard*index: shard*index + shard], x)
        req = dist.isend(tensor=out, dst=pseudo_master)
        shout("started sending "+str(out))
        req.wait()
    
    else:
        pass


def net():
        master = 0
        W = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.],
                      [1., 2., 3.], [4., 5., 6.], [7., 8., 9.],
                      [1., 2., 3.], [4., 5., 6.], [7., 8., 9.],
                      [1., 2., 3.], [4., 5., 6.], [7., 8., 9.],
                      [1., 2., 3.], [4., 5., 6.], [7., 8., 9.],
                      [1., 2., 3.], [4., 5., 6.], [7., 8., 9.],
                      [1., 2., 3.], [4., 5., 6.]])
        W = torch.tensor(W)

        x = np.array([[3., 3.], [1., 1.], [4., 4.]])
        x = torch.tensor(x)

        assignments = np.array([master, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        drop = 0.5
        y1 = layer(W, x, assignments, drop)
        assignments = np.array([master, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
        y2 = layer(W, x, assignments, drop)
        if(dist.get_rank() == assignments[0]):
            print(y1)
            print(y2)


def init_processes(fn, backend='mpi'):
    """ Initialize the distributed environment. """
    #os.environ['MASTER_ADDR'] = '127.0.0.1'
    #os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(init_method='env://', backend='mpi')
    fn()

if __name__ == "__main__":
    init_processes(net, backend='mpi')
