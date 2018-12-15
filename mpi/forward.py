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

	shut = np.random.random_sample()
	if(shut > 0.8):
		#print(str(dist.get_rank())+":Failure..")
		time.sleep(5)
		#print(str(dist.get_rank())+":Revived")


def shout(msg):
	if(DEBUG):
		print("Node " + str(dist.get_rank()) + " " + str(getAddress()) + " " + str(msg))

def layer(W, x, assignments, drop):
	t1 = time.time()
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
		#print("Waiting")
		req.wait()
		shout("Received top 50 percent")
		t2 = time.time()
		#print(t2 - t1)
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

		#shout(y2)
		shout("Gathered top 50 percent ")
		shout("Sending top 50 percent")
		#print(recvd)
		req = dist.isend(tensor=y2, dst=master)
		req.wait()

		for i in range(num_slaves):
			reqPieces[i].wait()

	
	elif(dist.get_rank() in slaves):
		straggler()

		index = list(slaves).index(dist.get_rank())
		out = torch.zeros(shard, p)
		out[:] = torch.matmul(W[shard*index: shard*index + shard], x)/(1 - drop)
		req = dist.isend(tensor=out, dst=pseudo_master)
		shout("started sending ")
		req.wait()
	
	else:
		pass


def net(drop):
	master = 0
	learning_rate = 1e-6
	
	W1 = torch.zeros(100, 1000)
	W2 = torch.zeros(10, 100)
	x = torch.zeros(1000, 64)

	if(dist.get_rank() == 0):
		W1 = torch.randn(100, 1000)
		W2 = torch.randn(10, 100)
		x = torch.randn(1000, 64)
		y = torch.randn(10, 64)

	for t in range(100):
	
		#group = dist.new_group([0, 1])
		dist.broadcast(W1, 0)
		dist.broadcast(x, 0)

		assignments = np.arange(0, 10 + 2)
		h = layer(W1, x, assignments, 0.2)
		if(dist.get_rank() == 0):
			h_relu = h.clamp(min=0)
		else:
			h_relu = torch.zeros(100, 64)

		dist.broadcast(W2, 0)
		dist.broadcast(h_relu, 0)

		y_pred = layer(W2, h_relu, assignments, 0)

		if(dist.get_rank() == 0):
			# Compute and print loss
			loss = (y_pred - y).pow(2).sum().item()
			print(t, math.log10(loss))

			# Backprop to compute gradients of w1 and w2 with respect to loss
			grad_y_pred = 2.0 * (y_pred - y)
			grad_w2 = grad_y_pred.mm(h_relu.t()) #h_relu.t().mm(grad_y_pred)
			grad_h_relu = W2.t().mm(grad_y_pred) #grad_y_pred.mm(w2.t())
			grad_h = grad_h_relu.clone()
			grad_h[h < 0] = 0
			grad_w1 = grad_h.mm(x.t()) #x.t().mm(grad_h)

			# Update weights using gradient descent
			W1 -= learning_rate * grad_w1
			W2 -= learning_rate * grad_w2


def init_processes(fn, backend='mpi'):
	""" Initialize the distributed environment. """
	#os.environ['MASTER_ADDR'] = '127.0.0.1'
	#os.environ['MASTER_PORT'] = '29500'
	dist.init_process_group(init_method='env://', backend='mpi')
	'''
	processes = []
	for i in range(0,10):
		p = Process(target = fn)
		processes.append(p)
		p.start()

	for p in processes:
		p.join()
	'''
	
	ITER = 1

	for i in range(ITER):
		fn(0.5)
	

if __name__ == "__main__":
	init_processes(net, backend='mpi')
