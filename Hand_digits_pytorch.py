import Hand_digits as hd
import torch
from torch.autograd import Variable
import time


def create_torch_sets():

	x, y = hd.loadlocal_mnist(
		images_path='/Users/andrea/Desktop/deeplearning.ai/MNIST_train_set',
		labels_path='/Users/andrea/Desktop/deeplearning.ai/MNIST_train_labels')

	w, z = hd.loadlocal_mnist(
		images_path='/Users/andrea/Desktop/deeplearning.ai/MNIST_dev_set',
		labels_path='/Users/andrea/Desktop/deeplearning.ai/MNIST_dev_labels')

	return (torch.from_numpy(x).float(), torch.from_numpy(y).float(),
	        torch.from_numpy(w).float(), torch.from_numpy(z).float())


def create_comp_graph(train, hidden_layers, classes, init='xavier'):
	all_layers = [train.shape[1]] + hidden_layers
	modules = []

	for i in range(len(all_layers) - 1):
		modules.append(torch.nn.Linear(all_layers[i], all_layers[i + 1]))
		modules.append(torch.nn.ReLU())

	modules.append(torch.nn.Linear(all_layers[-1], classes))
	modules.append(torch.nn.Softmax(dim=1))

	# for j in range(0, len(hd.parameters) // 2):
	# 	w = hd.parameters['W{}'.format(j + 1)].T
	# 	b = hd.parameters['b{}'.format(j + 1)]
	# 	w = torch.nn.Parameter(torch.from_numpy(w).float())
	# 	b = torch.nn.Parameter(torch.from_numpy(b).float())
	#
	# 	modules[j * 2].weight = w
	# 	modules[j * 2].bias = b

	if init == 'xavier':
		for j in range(0, len(modules), 2):
			torch.nn.init.xavier_uniform(modules[j].weight)

	return torch.nn.Sequential(*modules)


def run_model(comp_graph, learning_rate, epochs, batch_size, cost_every):

	x = Variable(train_set)
	y = Variable(train_lab, requires_grad=False).long()
	loss = 0

	loss_fn = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	for i in range(epochs + 1):
		y_pred = comp_graph(x)
		loss = loss_fn(y_pred, y)
		if type(cost_every) == int and i % cost_every == 0:
			print('Cost at iteration {}: {}'.format(i, round(loss.data[0], 4)))
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	if cost_every == 'last':
		print('Cost at last iteration: {}'.format(round(loss.data[0], 4)))


train_set, train_lab, dev_set, dev_lab = create_torch_sets()

hid_lay = [32, 32, 32]
ny = 10

model = create_comp_graph(train_set, hid_lay, ny, init='xavier')

run_model(model, learning_rate=5e-4, epochs=1, batch_size=5, cost_every=1)
