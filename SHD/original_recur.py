#implementing recurrent spiking neural network as described in Surrogate Gradient Learning in Spiking Neural Networks. https://arxiv.org/abs/1901.09948
#using LIF neurons, setting hyperparameters.
#number of neurons in each layer N = 128
#learning rate alpha = 0.001
#betas for Adam optimizer, first and second momemt = 0.9, 0.999
#batch size for minibatch Nbatch = 256
#threshold potential Uthres = 1
#reset potential Urest = 0
#synaptic time constant tausyn = 10 ms = 0.01 s
#membrane time constant taumem = 20 ms = 0.02 s
#let the exponential form of time constants be lambd and muh
#refractory time constant tref = 0 ms
#time step size = 0.5 ms = 0.0005 s
#total simulation duration = 1s

import os
import h5py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

import torch
import torch.nn as nn
import torchvision
from torch.utils import data
import pickle


basepath = os.getcwd() 

direc = "/dataset/shddataset/"
print(basepath+direc)

train_file = h5py.File((basepath + direc +  "shd_train.h5"), 'r')
test_file = h5py.File((basepath +direc + "shd_test.h5"), 'r')


x_train = train_file['spikes']
y_train = train_file['labels']
x_test = test_file['spikes']
y_test = test_file['labels']


nb_inputs  = 700
nb_hidden  = 200
nb_outputs = 20

time_step = 1e-3
nb_steps = 100
max_time = 1.4

batch_size = 256

datatype = torch.float
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sparse_data_generator_from_hdf5_spikes(X, y, batch_size, nb_steps, nb_units, max_time, shuffle=True):
	""" This generator takes a spike dataset and generates spiking network input as sparse tensors. 

	Args:
		X: The data ( sample x event x 2 ) the last dim holds (time,neuron) tuples
		y: The labels"""

	labels_ = np.array(y,dtype=np.int)
	number_of_batches = len(labels_)//batch_size
	sample_index = np.arange(len(labels_))

	# compute discrete firing times
	firing_times = X['times']
	units_fired = X['units']

	time_bins = np.linspace(0, max_time, num=nb_steps)

	if shuffle:
		np.random.shuffle(sample_index)

	total_batch_count = 0
	counter = 0
	while counter<number_of_batches:
		batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]

		coo = [ [] for i in range(3) ]
		for bc,idx in enumerate(batch_index):
			times = np.digitize(firing_times[idx], time_bins)
			units = units_fired[idx]
			batch = [bc for _ in range(len(times))]
			coo[0].extend(batch)
			coo[1].extend(times)
			coo[2].extend(units)

		i = torch.LongTensor(coo).to(device)
		v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)

		X_batch = torch.sparse.FloatTensor(i, v, torch.Size([batch_size,nb_steps,nb_units])).to(device)
		y_batch = torch.tensor(labels_[batch_index],device=device)

		yield X_batch.to(device=device), y_batch.to(device=device)

		counter += 1


tau_mem = 10e-3
tau_syn = 5e-3

alpha   = float(np.exp(-time_step/tau_syn))
beta    = float(np.exp(-time_step/tau_mem))
weight_scale = 0.2

w1 = torch.empty((nb_inputs, nb_hidden),  device=device, dtype=datatype, requires_grad=True)
w2 = torch.empty((nb_hidden, nb_outputs), device=device, dtype=datatype, requires_grad=True)
v1 = torch.empty((nb_hidden, nb_hidden), device=device, dtype=datatype, requires_grad=True)
loss_hist = []

print("init done")

if os.path.isfile((basepath+"/trained_values/zenketrainedogrecure200NEW.pt")):
	print('The file is present.')
	w1,w2,v1 = torch.load(basepath+'/trained_values/zenketrainedogrecure200NEW.pt')
else:
	torch.nn.init.normal_(w1, mean=0.0, std=weight_scale/np.sqrt(nb_inputs))
	torch.nn.init.normal_(w2, mean=0.0, std=weight_scale/np.sqrt(nb_hidden))
	torch.nn.init.normal_(v1, mean=0.0, std=weight_scale/np.sqrt(nb_hidden))




class SurrGradSpike(torch.autograd.Function):
	scale = 100.0 # controls steepness of surrogate gradient
	@staticmethod
	def forward(ctx, input):
		ctx.save_for_backward(input)
		out = torch.zeros_like(input)
		out[input > 0] = 1.0
		return out
	@staticmethod
	def backward(ctx, grad_output):
		input, = ctx.saved_tensors
		grad_input = grad_output.clone()
		grad = grad_input/(SurrGradSpike.scale*torch.abs(input)+1.0)**2
		return grad
spike_fn  = SurrGradSpike.apply



def run_snn(inputs):
	syn = torch.zeros((batch_size,nb_hidden), device=device, dtype=datatype)
	mem = torch.zeros((batch_size,nb_hidden), device=device, dtype=datatype)

	mem_rec = []
	spk_rec = []

	out = torch.zeros((batch_size, nb_hidden), device=device, dtype=datatype)
	h1_from_input = torch.einsum("abc,cd->abd", (inputs, w1))
	for t in range(nb_steps):
		h1 = h1_from_input[:,t] + torch.einsum("ab,bc->ac", (out, v1))
		mthr = mem-1.0 #1 is the threshold potential
		out = spike_fn(mthr)
		rst = out.detach() # We do not want to backprop through the reset

		new_syn = alpha*syn +h1
		new_mem =(beta*mem +syn)*(1.0-rst)

		mem_rec.append(mem)
		spk_rec.append(out)
		mem = new_mem
		syn = new_syn

	mem_rec = torch.stack(mem_rec,dim=1)
	spk_rec = torch.stack(spk_rec,dim=1)

	# Readout layer
	h2= torch.einsum("abc,cd->abd", (spk_rec, w2))
	flt = torch.zeros((batch_size,nb_outputs), device=device, dtype=datatype)
	out = torch.zeros((batch_size,nb_outputs), device=device, dtype=datatype)
	out_rec = [out]
	for t in range(nb_steps):
		new_flt = alpha*flt +h2[:,t]
		new_out = beta*out +flt

		flt = new_flt
		out = new_out

		out_rec.append(out)

	out_rec = torch.stack(out_rec,dim=1)
	other_recs = [mem_rec, spk_rec]
	return out_rec, other_recs




def train(x_data, y_data, lr=1e-3, nb_epochs=10):

	params = [w1,w2,v1]
	optimizer = torch.optim.Adamax(params, lr=lr, betas=(0.9,0.999))

	log_softmax_fn = nn.LogSoftmax(dim=1)
	loss_fn = nn.NLLLoss()

	loss_hist = []
	for e in range(nb_epochs):
		local_loss = []
		for x_local, y_local in sparse_data_generator_from_hdf5_spikes(x_data, y_data, batch_size, nb_steps, nb_inputs, max_time):
			output,recs = run_snn(x_local.to_dense())
			_,spks=recs
			m,_=torch.max(output,1)
			log_p_y = log_softmax_fn(m)

			# Here we set up our regularizer loss
			# The strength paramters here are merely a guess and there should be ample room for improvement by
			# tuning these paramters.
			reg_loss = 2e-6*torch.sum(spks) # L1 loss on total number of spikes
			reg_loss += 2e-6*torch.mean(torch.sum(torch.sum(spks,dim=0),dim=0)**2) # L2 loss on spikes per neuron

			# Here we combine supervised loss and the regularizer
			y_local = y_local.type(torch.cuda.LongTensor)
			loss_val = loss_fn(log_p_y, y_local) + reg_loss

			optimizer.zero_grad()
			loss_val.backward()
			optimizer.step()
			local_loss.append(loss_val.item())
		mean_loss = np.mean(local_loss)
		loss_hist.append(mean_loss)
		print("Epoch %i: loss=%.5f"%(e+1,mean_loss))
	return loss_hist,params

def compute_classification_accuracy(x_data, y_data):
	""" Computes classification accuracy on supplied data in batches. """
	accs = []
	for x_local, y_local in sparse_data_generator_from_hdf5_spikes(x_data, y_data, batch_size, nb_steps, nb_inputs, max_time, shuffle=False):
		output,_ = run_snn(x_local.to_dense())
		m,_= torch.max(output,1) # max over time
		_,am=torch.max(m,1)      # argmax over output units
		tmp = np.mean((y_local==am).detach().cpu().numpy()) # compare to labels
		accs.append(tmp)
	return np.mean(accs)



nb_epochs = 500
loss_list,[rw1,rw2,rv1] = train(x_train, y_train, lr=2e-4, nb_epochs=nb_epochs)
print("Training accuracy: %.3f"%(compute_classification_accuracy(x_train,y_train)))
print("Test accuracy: %.3f"%(compute_classification_accuracy(x_test,y_test)))
loss_hist.append(loss_list)


torch.save([rw1,rw2,rv1] , basepath+'/trained_values/zenketrainedogrecure200NEW.pt')
open_file = open((basepath+"/trained_values/zenketrainedhistogrecur200NEW.pkl"),"ab")
pickle.dump(loss_list,open_file)
print("\nFile list dumped\n")
open_file.close()
print("Saved in file")
