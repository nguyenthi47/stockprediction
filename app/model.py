import torch
import torch.nn as nn

# input seq, batch, input_size
# seq itself, mini-batch, indexes elements if the input
# output seq, batch, hidden_size
class LSTM(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, output_dim):
		super(LSTM,self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True) #(batch, seq, feature)
		self.fc = nn.Linear(hidden_size, output_dim)

	def forward(self,x):
		# at default h_0, c_0 is zero, using require_grad() to update gradient by back propagation
		h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
		c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
		output, (hn,cn) = self.lstm(x, (h_0.detach(),c_0.detach()))
		output = self.fc(output[:,-1,:])
		return output





